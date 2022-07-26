from tensorflow.keras.optimizers import Adam

from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras import backend as kb
from scipy.stats import gaussian_kde
from IPython import display
import matplotlib.pyplot as plt
import keras as krs
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


class GAN():
    def __init__(self, x_train, out_file = 'lows.txt'):
        self.X_train = x_train
        self.out_file = out_file
        self.mol_shape = (self.X_train.shape[1],)
        self.latent_dim = 2
        optimizer = Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        mol_inp = Input(shape=self.mol_shape)
        low = self.encoder(mol_inp)
        mol_out = self.decoder(low)
        self.autoencoder = Model(mol_inp, mol_out)
        self.autoencoder.compile(loss='mean_squared_error', optimizer=optimizer)
        validity = self.discriminator(low)
        self.combined = Model(mol_inp, validity)
        self.combined.compile(loss='mean_squared_error', optimizer=optimizer)
        
        
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
        model.add(Dense(1, activation='sigmoid'))
        model.summary(print_fn=logging.info)
        mol = Input(shape=(self.latent_dim,))
        validity = model(mol)
        return Model(mol, validity)


    def train(self, epochs, batch_size=128, visualize_freq=200): 
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, self.X_train.shape[0], batch_size)
            mols = self.X_train[idx]
            gen_lows = self.encoder.predict(mols)
            gen_mols = self.decoder.predict(gen_lows)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            idx = np.random.randint(0, self.X_train.shape[0], batch_size)
            mols = self.X_train[idx]
            ae_loss = self.autoencoder.train_on_batch(mols, mols)
            c_loss = self.combined.train_on_batch(mols, valid)
            d_loss_real = self.discriminator.train_on_batch(noise, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_lows, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            if epoch % 100 == 0:
                output = f"{epoch} [D loss: {d_loss[0]},acc.: {100*d_loss[1]}]" + \
                      f"[AE loss: {ae_loss}] [C loss: {c_loss}]"
                logging.info(output)
            
            if epoch % visualize_freq == 0:
                tmplows = self.encoder(self.X_train)
                np.savetxt(f'{os.path.expanduser("~/visualization")}' + '/tmplows.txt', tmplows)
                self._make_visualization(f'{os.path.expanduser("~/visualization")}' + '/tmplows.txt')
                plt.pause(0.01)
        
        newlows = self.encoder(self.X_train)
        np.savetxt(self.out_file, newlows)

        