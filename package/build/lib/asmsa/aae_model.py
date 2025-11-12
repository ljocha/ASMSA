#! vim: ai expandtab ts=4:
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_probability as tfp
import math

from .utils import (
    _compute_number_of_neurons, _default_hp, _losses, 
    _Prior, _PriorDistribution, _PriorMultivariate, _PriorImage,
    _random_init, _SparseConstraint, _SparseInitializer, _masks
)


class AAEModel(keras.models.Model):
    def __init__(self, molecule_shape, latent_dim=2,
                 enc_layers=None, enc_seed=None,
                 disc_layers=None, disc_seed=None,
                 prior=tfp.distributions.Normal(loc=0, scale=1), hp=_default_hp, with_density=False,
                 with_cv1_bias=False):
        super().__init__()

        if enc_layers is None: enc_layers = hp['ae_number_of_layers']
        if disc_layers is None: disc_layers = hp['disc_number_of_layers']
        if enc_seed is None: enc_seed = hp['ae_neuron_number_seed']
        if disc_seed is None: disc_seed = hp['disc_neuron_number_seed']


        self.hp = hp
        self.latent_dim = latent_dim
        if isinstance(prior, tfp.distributions.Distribution):
            if len(prior.event_shape) == 0:
                self.get_prior = _PriorDistribution(latent_dim, prior)
            else:
                assert latent_dim == prior.event_shape[0]
                self.get_prior = _PriorMultivariate(prior)
        else:
            self.get_prior = _PriorImage(latent_dim, prior)

        self.with_density = with_density
        self.with_cv1_bias = with_cv1_bias

        assert not (with_density and with_cv1_bias)

        self.enc_seed = enc_seed
        self.disc_seed = disc_seed

        if not isinstance(enc_seed, list):
            enc_seed = [enc_seed]

        if not isinstance(disc_seed, list):
            disc_seed = [disc_seed]

        enc_neurons = np.array([_compute_number_of_neurons(enc_layers, n) for n in enc_seed])
        disc_neurons = np.array([_compute_number_of_neurons(disc_layers, n) for n in disc_seed])

        self.n_models = enc_neurons.shape[0]
        assert disc_neurons.shape[0] == self.n_models

        if with_density:
            assert self.n_models == 1  # don't bother with multiple models and density together
            self.prior_max = np.max(prior.prob(prior.sample(10000)))

        inp = keras.Input(shape=molecule_shape)
        out = inp

        out = keras.layers.Dense(np.sum(enc_neurons[:, 0]), activation=hp['activation'], name='enc_0')(out)
        out = keras.layers.BatchNormalization(momentum=0.8, name=f'enc_bn_0')(out)

        for num in range(1, enc_neurons.shape[1]):
            name = f'enc_{num}'
            out = keras.layers.Dense(np.sum(enc_neurons[:, num]), activation=hp['activation'],
                                     name=name, **_masks(enc_neurons[:, num - 1], enc_neurons[:, num]))(out)
            out = keras.layers.BatchNormalization(momentum=0.8, name=f'enc_bn_{num}')(out)

        out = keras.layers.Dense(self.n_models * latent_dim, name='enc_out',
                                 **_masks(enc_neurons[:, -1], [latent_dim] * self.n_models))(out)
        latent = out

        out = keras.layers.Dense(np.sum(enc_neurons[:, -1]), activation=hp['activation'],
                                 name=f'dec_{enc_neurons.shape[1]}',
                                 **_masks([latent_dim] * self.n_models, enc_neurons[:, -1]))(out)
        out = keras.layers.BatchNormalization(momentum=0.8, name=f'dec_bn_{enc_neurons.shape[1]}')(out)

        # decoder layers are numbered in reverse so that neuron numbers match with encoder
        for num in reversed(range(enc_neurons.shape[1] - 1)):
            name = f'dec_{num}'
            out = keras.layers.Dense(np.sum(enc_neurons[:, num]), activation=hp['activation'], name=name,
                                     **_masks(enc_neurons[:, num + 1], enc_neurons[:, num]))(out)
            out = keras.layers.BatchNormalization(momentum=0.8, name=f'dec_bn_{num}')(out)

        out = keras.layers.Dense(self.n_models * molecule_shape[0], name='dec_out', activation=None, #hp['activation'],
                                 **_masks(enc_neurons[:, 0], [molecule_shape[0]] * self.n_models))(out)
        out = keras.layers.Reshape((self.n_models, molecule_shape[0]))(out)

        self.aes = keras.Model(inputs=inp, outputs=[out, latent])
        self.enc = keras.Model(inputs=inp, outputs=latent)
        self.dec = keras.Model(inputs=latent, outputs=out)

        inp = keras.Input(shape=(latent_dim * self.n_models,))
        disc = inp
        disc = keras.layers.Dense(np.sum(disc_neurons[:, 0]), name='disc_0',
                                  **_masks([latent_dim] * self.n_models, disc_neurons[:, 0]))(disc)
        disc = keras.layers.LeakyReLU(alpha=0.2, name=f'disc_relu_{num}')(disc)

        for num in range(1, disc_neurons.shape[1]):
            name = f'disc_{num}'
            disc = keras.layers.Dense(np.sum(disc_neurons[:, num]), name=name,
                                      **_masks(disc_neurons[:, num - 1], disc_neurons[:, num]))(disc)
            disc = keras.layers.LeakyReLU(alpha=0.2, name=f'disc_relu_{num}')(disc)

        disc = keras.layers.Dense(self.n_models, name='disc_out',
                                  **_masks(disc_neurons[:, -1], [1] * self.n_models))(disc)

        self.disc = keras.Model(inputs=inp, outputs=disc)

    def compile(self, optimizer=None, ae_loss=None):
        if optimizer is None:
            optimizer = self.hp['optimizer']

        if isinstance(optimizer, str):
            #            optimizer = keras.optimizers.legacy.__dict__[optimizer]
            optimizer = keras.optimizers.__dict__[optimizer]

        self.ae_loss_fn = _losses[ae_loss if ae_loss else self.hp['ae_loss_fn']]
        self.dens_loss_fn = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)

        super().compile(optimizer=optimizer(learning_rate=self.hp['learning_rate']), loss=self.ae_loss_fn)
        self.ae_weights = self.enc.trainable_weights + self.dec.trainable_weights
        if self.disc is not None:
            self.optimizer.build(self.ae_weights + self.disc.trainable_weights)
        else:
            self.optimizer.build(self.ae_weights)

        self.enc.compile(loss=self.ae_loss_fn)
        self.dec.compile(loss=self.ae_loss_fn)

    @tf.function
    def train_step(self, in_batch):
        if isinstance(in_batch, tuple):
            batch = in_batch[0]
        else:
            batch = in_batch

        # multiple models need replicated batch to compute loss simultaneously
        multibatch = tf.stack([batch] * self.n_models, axis=1)

        # AUTOENCODER
        with tf.GradientTape() as aetape:
            reconstruct = self.aes(batch)
            mse = self.ae_loss_fn(multibatch, reconstruct[0])
            ae_multiloss = tf.reduce_mean(mse, axis=0)
            ae_loss = tf.reduce_sum(ae_multiloss)

        ae_grad = aetape.gradient(ae_loss, self.ae_weights)
        self.last_ae_grad = ae_grad
        self.optimizer.apply_gradients(zip(ae_grad, self.ae_weights))

        rand_low = self.get_prior((batch.shape[0],))
        rand_low = tf.tile(rand_low, (1, self.n_models))

        # DISCRIMINATOR
        # XXX: Binary crossentropy from logits hardcoded
        disc_losses = tf.constant([0.])
        cheat_losses = tf.constant([0.])
        if self.disc is not None:
            with tf.GradientTape() as dtape:
                neg_pred = self.disc(reconstruct[1])
                neg_losses = tf.reduce_mean(neg_pred * tf.random.uniform(tf.shape(neg_pred), 1., 1.05), axis=0)
                pos_pred = self.disc(rand_low)
                pos_losses = -tf.reduce_mean(pos_pred * tf.random.uniform(tf.shape(pos_pred), 1., 1.05), axis=0)
                disc_losses = neg_losses + pos_losses
                disc_loss = tf.reduce_mean(disc_losses)

            disc_grads = dtape.gradient(disc_loss, self.disc.trainable_weights)
            self.optimizer.apply_gradients(zip(disc_grads, self.disc.trainable_weights))

            # dtto
            # CHEAT DISCRIMINATOR
            with tf.GradientTape() as ctape:
                cheat = self.disc(self.enc(batch))
                cheat_losses = -tf.reduce_mean(cheat * tf.random.uniform(tf.shape(cheat), 1., 1.05), axis=0)
                cheat_loss = tf.reduce_mean(cheat_losses)

            cheat_grads = ctape.gradient(cheat_loss, self.enc.trainable_weights)
            self.optimizer.apply_gradients(zip(cheat_grads, self.enc.trainable_weights))

        dens_loss = 42.

        # FOLLOW DENSITIES
        if self.with_density:
            with tf.GradientTape() as detape:
                lows = self.enc(batch)
                low_dens = self.get_prior.prior.prob(lows)  # XXX assumes MultivariateNormal, more or less
                low_dens /= self.prior_max
                # dens_loss = keras.losses.kl_divergence(in_batch[1],low_dens)
                # dens_loss = keras.losses.mean_squared_error(in_batch[1],low_dens)
                dens_loss = self.dens_loss_fn(in_batch[1], low_dens)

            dens_grads = detape.gradient(dens_loss, self.enc.trainable_weights)
            self.optimizer.apply_gradients(zip(dens_grads, self.enc.trainable_weights))

        # BIAS CV1
        if self.with_cv1_bias:
            with tf.GradientTape() as btape:
                lows = self.enc(batch)
                dens_loss = self.dens_loss_fn(in_batch[1], lows[:, 0])

            bias_grads = btape.gradient(dens_loss, self.enc.trainable_weights)
            self.optimizer.apply_gradients(zip(bias_grads, self.enc.trainable_weights))

        return {
            'AE loss min': tf.reduce_min(ae_multiloss),
            'disc loss min': tf.reduce_min(disc_losses),
        }

    @tf.function
    def test_step(self, in_batch):
        if isinstance(in_batch, tuple):
            batch = in_batch[0]
        else:
            batch = in_batch

        # Multiple models need replicated batch to compute loss simultaneously
        multibatch = tf.stack([batch] * self.n_models, axis=1)

        # AUTOENCODER - compute reconstruction loss
        reconstruct = self.aes(batch)
        mse = self.ae_loss_fn(multibatch, reconstruct[0])
        ae_multiloss = tf.reduce_mean(mse, axis=0)
        ae_loss = tf.reduce_sum(ae_multiloss)

        # Get prior samples for discriminator validation
        rand_low = self.get_prior((batch.shape[0],))
        rand_low = tf.tile(rand_low, (1, self.n_models))

        # DISCRIMINATOR - compute validation metrics without updating
        disc_losses = tf.constant([0.])
        cheat_losses = tf.constant([0.])
        if self.disc is not None:
            neg_pred = self.disc(reconstruct[1])
            neg_losses = tf.reduce_mean(neg_pred * tf.random.uniform(tf.shape(neg_pred), 1., 1.05), axis=0)
            pos_pred = self.disc(rand_low)
            pos_losses = -tf.reduce_mean(pos_pred * tf.random.uniform(tf.shape(pos_pred), 1., 1.05), axis=0)
            disc_losses = neg_losses + pos_losses

            # CHEAT DISCRIMINATOR
            cheat = self.disc(self.enc(batch))
            cheat_losses = -tf.reduce_mean(cheat * tf.random.uniform(tf.shape(cheat), 1., 1.05), axis=0)

        # Density validation
        dens_loss = 42.
        if self.with_density and isinstance(in_batch, tuple) and len(in_batch) > 1:
            lows = self.enc(batch)
            low_dens = self.get_prior.prior.prob(lows)
            low_dens /= self.prior_max
            dens_loss = self.dens_loss_fn(in_batch[1], low_dens)

        # Bias CV1 validation
        if self.with_cv1_bias and isinstance(in_batch, tuple) and len(in_batch) > 1:
            lows = self.enc(batch)
            dens_loss = self.dens_loss_fn(in_batch[1], lows[:, 0])

        return {
            'AE loss min': tf.reduce_min(ae_multiloss),
            'disc loss min': tf.reduce_min(disc_losses),
        }

    def summary(self, expand_nested=True):
        self.aes.summary(expand_nested=expand_nested)
        self.disc.summary(expand_nested=expand_nested)

    @tf.function
    def call(self, x, **kwargs):
        return self.dec(self.enc(x))

    @tf.function
    def call_enc(self, x):
        return self.enc(x)

    @tf.function
    def call_disc(self, low):
        return self.disc(low)


class GaussianMixture(tfp.distributions.MultivariateNormalDiag):
    def __init__(self, means, devs, weights):
        super().__init__(loc=[0., 0.])  # XXX
        self.dists = [tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=dev) for loc, dev in
                      zip(means, devs)]
        self.weights = weights
        assert sum(weights) == 1.0

    def sample(self, shape):
        if isinstance(shape, int): shape = (shape,)
        flat = math.prod(shape)
        nsamples = [int(flat * w) for w in self.weights]
        nsamples[0] += flat - sum(nsamples)

        samples = [d.sample((n,)) for d, n in zip(self.dists, nsamples)]
        return tf.reshape(tf.concat(samples, axis=0), (*shape, 2))  # XXX

    def prob(self, sample):
        probs = [w * d.prob(sample) for w, d in zip(self.weights, self.dists)]
        return tf.math.reduce_sum(tf.stack(probs, axis=0), axis=0)