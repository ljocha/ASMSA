from tensorflow.keras.optimizers import Adam

from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras import backend as kb
import keras as krs
import numpy as np
import logging


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


    def train(self, epochs, batch_size=128): 
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
        newlows = self.encoder(self.X_train)
        np.savetxt(self.out_file, newlows)
