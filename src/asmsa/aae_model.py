#! vim: ai expandtab ts=4:

import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL
import tensorflow_probability as tfp
import math

def _compute_number_of_neurons(layers,seed):
    neurons = [seed]
    tmp = seed
    for _ in range(layers):
        tmp = int(tmp / 2)
        neurons.append(tmp)
    return neurons



_default_hp = {
    'activation' : 'gelu',
    'ae_loss_fn': 'MeanSquaredError',
    'optimizer': 'Adam',
    'learning_rate' : 0.0002
}


_losses = {
    'MeanSquaredError' : keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE),
    'Huber' :keras.losses.Huber(reduction=keras.losses.Reduction.NONE),
}

class _Prior():
    def __init__(self,latent_dim):
        self.latent_dim = latent_dim

    def __call__(self,shape):
        pass

class _PriorDistribution(_Prior):
    def __init__(self,latent_dim,prior):
        super().__init__(latent_dim)
        self.prior = prior
    
    @tf.function
    def __call__(self,shape):
        sample = tfp.distributions.Sample(
            self.prior,
            sample_shape=(*shape,self.latent_dim)).sample()

        if len(sample.shape) > 2:
            sample = tf.reshape(sample,[sample.shape[0]*sample.shape[1],self.latent_dim])
            
        return sample

class _PriorMultivariate(_Prior):
    def __init__(self,prior):
        super().__init__(prior.event_shape[0])
        self.prior = prior

    @tf.function
    def __call__(self,shape):
      return self.prior.sample(shape)


class _PriorImage(_Prior):
    def __init__(self,latent_dim,file):
        super().__init__(latent_dim)
        img = np.array(PIL.Image.open(file),dtype=np.float64)
        if len(img.shape) == 3:
            img = np.sum(img,axis=2)

        assert len(img.shape) == 2 and latent_dim == 2

        img /= np.sum(img)
        self.shape = img.shape
        self.flat = tf.convert_to_tensor(np.cumsum(img.flatten()).astype(np.float32))

    @tf.function
    def __call__(self,shape):
        u = tf.random.uniform(shape=shape)
        b = tf.broadcast_to(self.flat,u.shape[:-1]+self.flat.shape)
        flati = tf.searchsorted(b,u)
        x = tf.cast(flati % self.shape[1],tf.float32)
        x += tf.random.uniform(x.shape)
        y = tf.cast(flati // self.shape[1],tf.float32)
        y += tf.random.uniform(y.shape)
        x /= self.shape[1]
        y = 1. - y/self.shape[0]
        return tf.stack([x,y],axis=-1)


_random_init = keras.initializers.GlorotUniform(seed=42)


class _SparseConstraint(keras.constraints.Constraint):
    def __init__(self,left,right):
        super().__init__()
        assert len(left) == len(right)
        mask = np.zeros((np.sum(left),np.sum(right)),dtype=np.float32)
        idxl = np.concatenate((np.zeros((1,),dtype=np.int32),np.cumsum(left)))
        idxr = np.concatenate((np.zeros((1,),dtype=np.int32),np.cumsum(right)))
        for mod in range(len(left)):
            mask[idxl[mod]:idxl[mod+1],idxr[mod]:idxr[mod+1]] = 1.

        self.mask = tf.convert_to_tensor(mask)

    def __call__(self,w):
        return w * self.mask

class _SparseInitializer(keras.initializers.Initializer):
    def __init__(self,left,right):
        super().__init__()
        assert len(left) == len(right)
        self.left = left
        self.right = right
        self.idxl = np.concatenate((np.zeros((1,),dtype=np.int32),np.cumsum(left)))
        self.idxr = np.concatenate((np.zeros((1,),dtype=np.int32),np.cumsum(right)))

    def __call__(self,shape,dtype=None):
        assert list(shape) == [np.sum(self.left),np.sum(self.right)]

        init = np.zeros((np.sum(self.left),np.sum(self.right)),dtype=dtype)
        for mod in range(len(self.left)):
            init[self.idxl[mod]:self.idxl[mod+1],self.idxr[mod]:self.idxr[mod+1]] = _random_init((self.left[mod],self.right[mod])).numpy()

        return tf.convert_to_tensor(init)


def _masks(left,right):
    return { 'kernel_initializer': _SparseInitializer(left,right),
            'kernel_constraint': _SparseConstraint(left,right) }


class AAEModel(keras.models.Model):
    def __init__(self,molecule_shape,latent_dim=2,
            enc_layers=None,enc_seed=None,
            disc_layers=None,disc_seed=None,
            prior=tfp.distributions.Normal(loc=0, scale=1),hp=_default_hp,with_density=False,with_cv1_bias=False):
        super().__init__()
        
        if enc_layers is None: enc_layers = hp['ae_number_of_layers']
        if disc_layers is None: disc_layers = hp['disc_number_of_layers']
        if enc_seed is None: enc_seed = hp['ae_neuron_number_seed']
        if disc_seed is None: enc_seed = hp['disc_neuron_number_seed']

        self.hp = hp
        self.latent_dim = latent_dim
        if isinstance(prior,tfp.distributions.Distribution):
            if len(prior.event_shape) == 0:
               self.get_prior = _PriorDistribution(latent_dim,prior)
            else:
               assert latent_dim == prior.event_shape[0]
               self.get_prior = _PriorMultivariate(prior)
        else: 
            self.get_prior = _PriorImage(latent_dim,prior)

        self.with_density = with_density
        self.with_cv1_bias = with_cv1_bias

        assert not (with_density and with_cv1_bias)
            
        self.enc_seed = enc_seed
        self.disc_seed = disc_seed
            
        if not isinstance(enc_seed,list):
            enc_seed = [enc_seed]

        if not isinstance(disc_seed,list):
            disc_seed = [disc_seed]

        enc_neurons = np.array([ _compute_number_of_neurons(enc_layers,n) for n in enc_seed ])
        disc_neurons = np.array([ _compute_number_of_neurons(disc_layers,n) for n in disc_seed ])

        self.n_models = enc_neurons.shape[0]
        assert disc_neurons.shape[0] == self.n_models

        if with_density:
          assert self.n_models == 1		# don't bother with multiple models and density together
          self.prior_max = np.max(prior.prob(prior.sample(10000)))

        inp = keras.Input(shape=molecule_shape)
        out = inp

        out = keras.layers.Dense(np.sum(enc_neurons[:,0]),activation=hp['activation'],name = 'enc_0')(out)
        out = keras.layers.BatchNormalization(momentum=0.8,name=f'enc_bn_0')(out)

        for num in range(1,enc_neurons.shape[1]):
            name = f'enc_{num}'
            out = keras.layers.Dense(np.sum(enc_neurons[:,num]),activation=hp['activation'],
                                     name = name, **_masks(enc_neurons[:,num-1],enc_neurons[:,num]))(out)
            out = keras.layers.BatchNormalization(momentum=0.8,name=f'enc_bn_{num}')(out)

            
        out = keras.layers.Dense(self.n_models*latent_dim,name='enc_out',
                                 **_masks(enc_neurons[:,-1],[latent_dim]*self.n_models))(out) 
        latent = out

        out = keras.layers.Dense(np.sum(enc_neurons[:,-1]),activation=hp['activation'],name=f'dec_{enc_neurons.shape[1]}',
                                 **_masks([latent_dim]*self.n_models,enc_neurons[:,-1]))(out)
        out = keras.layers.BatchNormalization(momentum=0.8,name=f'dec_bn_{enc_neurons.shape[1]}')(out)

        # decoder layers are numbered in reverse so that neuron numbers match with encoder
        for num in reversed(range(enc_neurons.shape[1]-1)):
            name = f'dec_{num}'
            out = keras.layers.Dense(np.sum(enc_neurons[:,num]),activation=hp['activation'],name=name,
                                     **_masks(enc_neurons[:,num+1],enc_neurons[:,num]))(out)
            out = keras.layers.BatchNormalization(momentum=0.8,name=f'dec_bn_{num}')(out)

        out = keras.layers.Dense(self.n_models*molecule_shape[0],name='dec_out',activation=hp['activation'],
                                    **_masks(enc_neurons[:,0],[molecule_shape[0]]*self.n_models))(out)
        out = keras.layers.Reshape((self.n_models,molecule_shape[0]))(out)

        self.aes = keras.Model(inputs=inp,outputs=[out,latent])
        self.enc = keras.Model(inputs=inp,outputs=latent)
        self.dec = keras.Model(inputs=latent,outputs=out)

        inp = keras.Input(shape=(latent_dim * self.n_models,))
        disc = inp
        disc = keras.layers.Dense(np.sum(disc_neurons[:,0]),name='disc_0',
                                  **_masks([latent_dim]*self.n_models,disc_neurons[:,0]))(disc)
        disc = keras.layers.LeakyReLU(alpha=0.2,name=f'disc_relu_{num}')(disc)

        for num in range(1,disc_neurons.shape[1]):
            name = f'disc_{num}'
            disc = keras.layers.Dense(np.sum(disc_neurons[:,num]),name=name,
                                      **_masks(disc_neurons[:,num-1],disc_neurons[:,num]))(disc)
            disc = keras.layers.LeakyReLU(alpha=0.2,name=f'disc_relu_{num}')(disc)

        disc = keras.layers.Dense(self.n_models,name='disc_out',
                                  **_masks(disc_neurons[:,-1],[1]*self.n_models))(disc)

        self.disc = keras.Model(inputs=inp,outputs=disc)

    def compile(self,optimizer=None,ae_loss=None):
        if optimizer is None:
            optimizer = self.hp['optimizer']

        if isinstance(optimizer,str):
#            optimizer = keras.optimizers.legacy.__dict__[optimizer]
            optimizer = keras.optimizers.__dict__[optimizer]

        super().compile(optimizer = optimizer(learning_rate=self.hp['learning_rate']))
        self.ae_weights = self.enc.trainable_weights + self.dec.trainable_weights
        if self.disc is not None:
            self.optimizer.build(self.ae_weights + self.disc.trainable_weights)
        else:
            self.optimizer.build(self.ae_weights)
        self.ae_loss_fn = _losses[ae_loss if ae_loss else self.hp['ae_loss_fn']]
        self.dens_loss_fn = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)


    @tf.function
    def train_step(self,in_batch):
        if isinstance(in_batch,tuple):
            batch = in_batch[0]
        else:
            batch = in_batch

        # multiple models need replicated batch to compute loss simultaneously
        multibatch = tf.stack([batch]*self.n_models,axis=1)

        #AUTOENCODER
        with tf.GradientTape() as aetape:
            reconstruct = self.aes(batch)
            mse = self.ae_loss_fn(multibatch,reconstruct[0])
            ae_multiloss = tf.reduce_mean(mse,axis=0)
            ae_loss = tf.reduce_sum(ae_multiloss)
            
        ae_grad = aetape.gradient(ae_loss,self.ae_weights)
        self.last_ae_grad = ae_grad
        self.optimizer.apply_gradients(zip(ae_grad,self.ae_weights))

        rand_low = self.get_prior((batch.shape[0],))
        rand_low = tf.tile(rand_low,(1,self.n_models))

        # DISCRIMINATOR
# XXX: Binary crossentropy from logits hardcoded
        disc_losses = tf.constant([0.])
        cheat_losses = tf.constant([0.])
        if self.disc is not None:
            with tf.GradientTape() as dtape:
                neg_pred = self.disc(reconstruct[1])
                neg_losses = tf.reduce_mean(neg_pred*tf.random.uniform(tf.shape(neg_pred),1.,1.05),axis=0) 
                pos_pred = self.disc(rand_low)
                pos_losses = -tf.reduce_mean(pos_pred*tf.random.uniform(tf.shape(pos_pred),1.,1.05),axis=0)
                disc_losses = neg_losses + pos_losses
                disc_loss = tf.reduce_mean(disc_losses)
    
            disc_grads = dtape.gradient(disc_loss,self.disc.trainable_weights)
            self.optimizer.apply_gradients(zip(disc_grads,self.disc.trainable_weights))
    
    # dtto
            # CHEAT DISCRIMINATOR
            with tf.GradientTape() as ctape:
                cheat = self.disc(self.enc(batch))
                cheat_losses = -tf.reduce_mean(cheat*tf.random.uniform(tf.shape(cheat),1.,1.05),axis=0)
                cheat_loss = tf.reduce_mean(cheat_losses)
    
            cheat_grads = ctape.gradient(cheat_loss,self.enc.trainable_weights)
            self.optimizer.apply_gradients(zip(cheat_grads,self.enc.trainable_weights))
    
        dens_loss = 42.
    
        # FOLLOW DENSITIES
        if self.with_density:
            with tf.GradientTape() as detape:
                 lows = self.enc(batch)
                 low_dens = self.get_prior.prior.prob(lows)	# XXX assumes MultivariateNormal, more or less
                 low_dens /= self.prior_max
                 #dens_loss = keras.losses.kl_divergence(in_batch[1],low_dens)
                 #dens_loss = keras.losses.mean_squared_error(in_batch[1],low_dens)
                 dens_loss = self.dens_loss_fn(in_batch[1],low_dens)

            dens_grads = detape.gradient(dens_loss,self.enc.trainable_weights)
            self.optimizer.apply_gradients(zip(dens_grads,self.enc.trainable_weights))

        # BIAS CV1
        if self.with_cv1_bias:
            with tf.GradientTape() as btape:
                lows = self.enc(batch)
                dens_loss = self.dens_loss_fn(in_batch[1],lows[:,0])
                            
            bias_grads = btape.gradient(dens_loss,self.enc.trainable_weights)
            self.optimizer.apply_gradients(zip(bias_grads,self.enc.trainable_weights))


        return {
            'AE loss min' : tf.reduce_min(ae_multiloss),
            'AE loss max' : tf.reduce_max(ae_multiloss),
            'disc loss min' : tf.reduce_min(disc_losses),
            'disc loss max' : tf.reduce_max(disc_losses),
            'cheat loss min' : tf.reduce_min(cheat_losses),
            'cheat loss max' : tf.reduce_max(cheat_losses),
            'density loss' : dens_loss
            }


    def summary(self,expand_nested=True):
        self.aes.summary(expand_nested=expand_nested)
        self.disc.summary(expand_nested=expand_nested)

    @tf.function
    def call(self,x,**kwargs):
        return self.dec(self.enc(x)) 

    @tf.function
    def call_enc(self,x):
        return self.enc(x)

    @tf.function
    def call_disc(self,low):
        return self.disc(low)


class GaussianMixture(tfp.distributions.MultivariateNormalDiag):
    def __init__(self,means,devs,weights):
        super().__init__(loc=[0.,0.]) # XXX
        self.dists = [ tfp.distributions.MultivariateNormalDiag(loc=loc,scale_diag=dev) for loc,dev in zip(means,devs) ]
        self.weights = weights
        assert sum(weights) == 1.0

    def sample(self,shape):
        if isinstance(shape,int): shape = (shape,)
        flat = math.prod(shape)
        nsamples = [ int(flat * w) for w in self.weights ]
        nsamples[0] += flat-sum(nsamples)

        samples = [ d.sample((n,)) for d,n in zip(self.dists,nsamples) ]
        return tf.reshape(tf.concat(samples,axis=0),(*shape,2)) # XXX

    def prob(self,sample):
        probs = [ w*d.prob(sample) for w,d in zip(self.weights,self.dists)]
        return tf.math.reduce_sum(tf.stack(probs,axis=0),axis=0)

