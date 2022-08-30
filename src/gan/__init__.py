import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import CSVLogger
from keras.models import Sequential, Model
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras import backend as kb
from scipy.stats import gaussian_kde
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os


class AAEModel(Model):
    def __init__(self,enc,dec,disc,lowdim,prior):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.disc = disc
        self.lowdim = lowdim
        self.prior = prior


    def compile(self,
        opt = Adam(0.0002,0.5),	# FIXME: justify
        ae_loss_fn = MeanSquaredError(),
# XXX: logits as in https://keras.io/guides/customizing_what_happens_in_fit/, 
# hope it works as the discriminator output is never used directly
        disc_loss_fn = BinaryCrossentropy(from_logits=True)	
    ):

        super().compile()
        self.opt = opt
        self.ae_loss_fn = ae_loss_fn
        self.disc_loss_fn = disc_loss_fn

        
    def train_step(self,batch):
        def _get_prior(name, shape):
            if name == "normal":
                return tf.random.normal(shape=shape)
            if name == "uniform":
                return tf.random.uniform(shape=shape)
            
            raise ValueError(f"Invalid prior type '{name}'. Choose from 'normal|uniform'")
        
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
        rand_low = _get_prior(self.prior, (batch_size, self.lowdim))
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
    def __init__(self, x_train, out_file = 'lows.txt', verbose=False, prior='normal'):
        self.X_train = x_train
        self.out_file = out_file
        self.mol_shape = (self.X_train.shape[1],)
        self.latent_dim = 2
        self.prior = prior
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.discriminator = self._build_discriminator()
        self.discriminator.compile(loss=BinaryCrossentropy(from_logits=True),
                                   optimizer=Adam(0.0002, 0.5),
                                   metrics=['accuracy'])
        self.discriminator.trainable = False

        self._compile(verbose)
        
        
    def _compile(self, verbose=False):
        self.aae = AAEModel(self.encoder,self.decoder,self.discriminator,self.latent_dim,self.prior)
        self.aae.compile()
        
        if verbose:
            print(self.encoder.summary(expand_nested=True))
            print(self.decoder.summary(expand_nested=True))
            print(self.discriminator.summary(expand_nested=True))

        
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
        
        
    def _build_encoder(self, params=[("selu", 32),
                                     ("selu", 16),
                                     ("selu", 8),
                                     ("linear", None)]):
        model = Sequential()
        # input layer
        model.add(Dense(params[0][1], input_dim=np.prod(self.mol_shape), activation=params[0][0]))
        model.add(BatchNormalization(momentum=0.8))
        # hidden layers
        model.add(Dense(params[1][1], activation=params[1][0]))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(params[2][1], activation=params[2][0]))
        model.add(BatchNormalization(momentum=0.8))
        #output layer
        model.add(Dense(self.latent_dim, activation=params[3][0]))
        mol = Input(shape=self.mol_shape)
        lowdim = model(mol)
        return Model(mol, lowdim, name="Encoder")
    
    
    def _build_decoder(self, params=[("selu", 8),
                                     ("selu", 16),
                                     ("selu", 32),
                                     ("linear", None)]):
        model = Sequential()
        model._name = "Decoder"
        # input layer
        model.add(Dense(params[0][1], input_dim=self.latent_dim, activation=params[0][0]))
        model.add(BatchNormalization(momentum=0.8))
        # hidden layers
        model.add(Dense(params[1][1], activation=params[1][0]))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(params[2][1], activation=params[2][0]))
        model.add(BatchNormalization(momentum=0.8))
        # output layer
        model.add(Dense(np.prod(self.mol_shape), activation=params[3][0]))
        model.add(Reshape(self.mol_shape))
        lowdim = Input(shape=(self.latent_dim,))
        mol = model(lowdim)
        return Model(lowdim, mol, name="Decoder")

    
    def _build_discriminator(self, params=[(None, 512),
                                           (None, 256),
                                           (None, 256),
                                           (None, 1)]):
        model = Sequential()
        model._name = "Discriminator"
        model.add(Flatten(input_shape=(self.latent_dim,)))
        model.add(Dense(params[0][1]))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(params[1][1]))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(params[2][1]))
        model.add(LeakyReLU(alpha=0.2))
# changed to match logit use in AAE.train_step()
#        model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(params[3][1]))

        mol = Input(shape=(self.latent_dim,))
        validity = model(mol)
        return Model(mol, validity, name="Discriminator")
    
    
    def set_encoder(self, params, build_decoder=False, verbose=False):
        model = self._build_encoder(params)
        self.encoder = model
        
        if build_decoder:
            # reverse parameters (output layer of decoder is the same as encoders')
            # e.g [1,2,3,4] -> [3,2,1,4]
            reversed_params = params[:-1][::-1] + params[-1:]
            self.set_decoder(reversed_params, verbose)
        self._compile()
        
        if verbose:
            print(self.encoder.summary(expand_nested=True))
        
        
    def set_decoder(self, params, verbose=False):
        model = self._build_decoder(params)
        self.decoder = model
        self._compile()
        
        if verbose:
            print(self.decoder.summary(expand_nested=True))
        
        
    def set_discriminator(self, params, verbose=False):
        model = self._build_discriminator(params)
        self.discriminator = model
        self._compile()
        
        if verbose:
            print(self.discriminator.summary(expand_nested=True))
        
    
    
        

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

        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        csv_logger = CSVLogger(logdir + 'log.csv', append=False, separator=';')

        callbacks = [csv_logger, tensorboard_callback]
        if visualize_freq:
            callbacks.append(GAN.VisualizeCallback(self,visualize_freq))
            
        self.aae.fit(dataset,epochs=epochs,verbose=True,callbacks=callbacks)

        newlows = self.encoder(self.X_train)
        np.savetxt(self.out_file, newlows)
