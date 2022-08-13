import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras import backend as kb
from scipy.stats import gaussian_kde
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import logging
import os


logging.root.handlers = []
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("gan.log", mode="w"),
                        logging.StreamHandler()
                    ])

def _normal_prior(shape):
	return tf.random.normal(shape=shape)

class AAEModel(Model):
	def __init__(self,enc,dec,disc,lowdim,prior = _normal_prior):
		super().__init__()
		self.enc = enc
		self.dec = dec
		self.disc = disc
		self.lowdim = lowdim
		self.prior = prior


	def compile(self,
		opt = Adam(0.0002,0.5),	# FIXME: justify
		ae_loss_fn = tf.keras.losses.MeanSquaredError(),
# XXX: logits as in https://keras.io/guides/customizing_what_happens_in_fit/, 
# hope it works as the discriminator output is never used directly
		disc_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)	
	):

		super().compile()
		self.opt = opt
		self.ae_loss_fn = ae_loss_fn
		self.disc_loss_fn = disc_loss_fn

	def train_step(self,batch):
		if isinstance(batch,tuple):
			batch = batch[0]

		batch_size = tf.shape(batch)[0]


# improve AE to reconstruct
		with tf.GradientTape(persistent=True) as ae_tape:
			reconstruct = self.dec(self.enc(batch))
			ae_loss = self.ae_loss_fn(batch,reconstruct)

		enc_grads = ae_tape.gradient(ae_loss, self.enc.trainable_weights)
		self.opt.apply_gradients(zip(enc_grads,self.enc.trainable_weights))

		dec_grads = ae_tape.gradient(ae_loss, self.dec.trainable_weights)
		self.opt.apply_gradients(zip(dec_grads,self.dec.trainable_weights))

# improve discriminator
		rand_low = self.prior((batch_size,self.lowdim))
		better_low = self.enc(batch)
		low = tf.concat([rand_low,better_low],axis=0)

		labels = tf.concat([tf.ones((batch_size,1)), tf.zeros((batch_size,1))], axis=0)
		labels += 0.05 * tf.random.uniform(tf.shape(labels))	# guide

		with tf.GradientTape() as disc_tape:
			pred = self.disc(low)
			disc_loss = self.disc_loss_fn(labels,pred)

		disc_grads = disc_tape.gradient(disc_loss,self.disc.trainable_weights)
		self.opt.apply_gradients(zip(disc_grads,self.disc.trainable_weights))

# teach encoder to cheat
		alltrue = tf.ones((batch_size,1))

		with tf.GradientTape() as cheat_tape:
			cheat = self.disc(self.enc(batch))
			cheat_loss = self.disc_loss_fn(alltrue,cheat)

		cheat_grads = cheat_tape.gradient(cheat_loss,self.enc.trainable_weights)
		self.opt.apply_gradients(zip(cheat_grads,self.enc.trainable_weights))

		return { 'ae_loss' : ae_loss, 'd_loss' : disc_loss }

		
		


class GAN():
    def __init__(self, x_train, out_file = 'lows.txt'):
        self.X_train = x_train
        self.out_file = out_file
        self.mol_shape = (self.X_train.shape[1],)
        self.latent_dim = 2
        self.discriminator = self.build_discriminator()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.aae = AAEModel(self.encoder,self.decoder,self.discriminator,self.latent_dim)
        self.aae.compile()

        
    def _make_visualization(self, output_file=None):
        if output_file is None:
            output_file = self.output_file
        
        os.chdir(os.path.expanduser("~/visualization"))
        lows = np.loadtxt(output_file)

        rama_ala = np.loadtxt('rama_ala_reduced.txt', usecols=(0,1))
        angever1 = np.loadtxt('angever1.txt')
        angever2 = np.loadtxt('angever2.txt')
        angever3 = np.loadtxt('angever3.txt')


        cvs = (lows[:, 0], lows[:, 1])
        analysis_files = {
            'rama0' : rama_ala[:, 0],
            'rama1' : rama_ala[:, 1],
            'ang1' : angever1[:, 1],
            'ang2' : angever2[:, 1],
            'ang3' : angever3[:, 1]
        }

        # set limits
        xmin, xmax = min(cvs[0]), max(cvs[0])
        ymin, ymax = min(cvs[1]), max(cvs[1])

        # plot configuration
        plt.suptitle('Low Dimentional Space - Analysis')
        plt.style.use("seaborn-white")
        fig = plt.figure(figsize=(18, 10))
        fig.supxlabel('CV1', x=0.5, fontsize=16, fontweight='bold')
        fig.supylabel('CV2', x=0.1, fontsize=16, fontweight='bold')

        # plot first graph
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        values = np.vstack([cvs[0], cvs[1]])
        kernel = gaussian_kde(values)
        dens = np.reshape(kernel(positions).T, X.shape)
        ax1 = plt.subplot(2, 3, 1)
        ax1.set_xticks([])
        plt.imshow(np.rot90(dens), cmap="hsv", aspect="auto", extent=[xmin, xmax, ymin, ymax])


        # plot every other graph
        i = 2
        for name, data in analysis_files.items():
            ax = plt.subplot(2, 3, i)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_title(name)
            if i in [2,3,5,6]:
                ax.set_yticks([])
            if i in [2,3]:
                ax.set_xticks([])
            plt.scatter(cvs[0], cvs[1], s=1, c=data, cmap="hsv")
            i += 1

        plt.savefig('analysis_tmp.png')
        os.chdir(os.path.expanduser("~"))
        
        
    def build_encoder(self):
        model = Sequential()
        model.add(Dense(256, input_dim=np.prod(self.mol_shape), activation="sigmoid"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim, activation='linear'))
        model.summary(print_fn=logging.info)
        mol = Input(shape=self.mol_shape)
        lowdim = model(mol)
        return Model(mol, lowdim)

    
    def build_decoder(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.latent_dim, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.mol_shape), activation='sigmoid'))
        model.add(Reshape(self.mol_shape))
        model.summary(print_fn=logging.info)
        lowdim = Input(shape=(self.latent_dim,))
        mol = model(lowdim)
        return Model(lowdim, mol)

    
    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=(self.latent_dim,)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
# changed to match logit use in AAE.train_step()
#        model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(1))
        model.summary(print_fn=logging.info)
        mol = Input(shape=(self.latent_dim,))
        validity = model(mol)
        return Model(mol, validity)

    class VisualizeCallback(tf.keras.callbacks.Callback):
        def __init__(self,parent,freq):
            super().__init__()
            self.parent = parent
            self.freq = freq

        def on_epoch_begin(self,epoch,logs=None):
            if epoch % self.freq == 0:
                tmplows = self.parent.encoder(self.parent.X_train)
                np.savetxt(f'{os.path.expanduser("~/visualization")}' + '/tmplows.txt', tmplows)
                self.parent._make_visualization(f'{os.path.expanduser("~/visualization")}' + '/tmplows.txt')
                plt.pause(0.01)
				    

    def train(self, epochs, batch_size=128, visualize_freq=False): 

        dataset = tf.data.Dataset.from_tensor_slices(self.X_train)
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

        callbacks = None
        if visualize_freq:
            callbacks = [GAN.VisualizeCallback(self,visualize_freq)]

        self.aae.fit(dataset,epochs=epochs,verbose=True,callbacks=callbacks)

        newlows = self.encoder(self.X_train)
        np.savetxt(self.out_file, newlows)
