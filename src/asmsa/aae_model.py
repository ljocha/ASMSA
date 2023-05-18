#! vim: ai ts=4:

import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL

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

class _PriorNormal(_Prior):
    @tf.function
    def __call__(self,shape):
        return tf.random.normal(shape=(*shape,self.latent_dim))

class _PriorUniform(_Prior):
    @tf.function
    def __call__(self,shape):
        return tf.random.uniform(shape=(*shape,self.latent_dim))

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
        assert len(left) == len(right)
        mask = np.zeros((np.sum(left),np.sum(right)),dtype=np.float32)
        idxl = np.concatenate((np.zeros((1,),dtype=np.int32),np.cumsum(left)))
        idxr = np.concatenate((np.zeros((1,),dtype=np.int32),np.cumsum(right)))
        for mod in range(len(left)):
            mask[idxl[mod]:idxl[mod+1],idxr[mod]:idxr[mod+1]] = 1.

        self.mask = tf.convert_to_tensor(mask)

    @tf.function	
    def __call__(self,w):
        return w * self.mask

class _SparseInitializer(keras.initializers.Initializer):
    def __init__(self,left,right):
        assert len(left) == len(right)
        self.left = left
        self.right = right
        self.idxl = np.concatenate((np.zeros((1,),dtype=np.int32),np.cumsum(left)))
        self.idxr = np.concatenate((np.zeros((1,),dtype=np.int32),np.cumsum(right)))

    def __call__(self,shape,dtype=None):
#		print(shape,self.left,self.right)
        assert shape == [np.sum(self.left),np.sum(self.right)]

        init = np.zeros((np.sum(self.left),np.sum(self.right)),dtype=dtype.as_numpy_dtype)
        for mod in range(len(self.left)):
            init[self.idxl[mod]:self.idxl[mod+1],self.idxr[mod]:self.idxr[mod+1]] = _random_init((self.left[mod],self.right[mod])).numpy()

        return tf.convert_to_tensor(init)


def _masks(left,right):
    return { 'kernel_initializer': _SparseInitializer(left,right),
            'kernel_constraint': _SparseConstraint(left,right) }


class AAEModel(keras.models.Model):
    def __init__(self,molecule_shape,latent_dim=2,
            enc_layers=2,enc_seed=64,
            disc_layers=2,disc_seed=64,
            prior='normal',hp=_default_hp):
        super().__init__()
        
        self.hp = hp
        self.latent_dim = latent_dim
        if prior == 'normal':
            self.get_prior = _PriorNormal(latent_dim)
        elif prior == 'uniform':
            self.get_prior = _PriorUniform(latent_dim)
        else: 
            self.get_prior = _PriorImage(latent_dim,prior)

            
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
            optimizer = keras.optimizers.legacy.__dict__[optimizer]

        super().compile(optimizer = optimizer(learning_rate=self.hp['learning_rate']))
        self.ae_weights = self.enc.trainable_weights + self.dec.trainable_weights
        self.ae_loss_fn = _losses[ae_loss if ae_loss else self.hp['ae_loss_fn']]

    @tf.function
    def train_step(self,batch):
        if isinstance(batch,tuple):
            batch = batch[0]

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

        rand_low = self.get_prior((tf.shape(batch)[0],))
        rand_low = tf.tile(rand_low,(1,self.n_models))

        # DISCRIMINATOR
# XXX: Binary crossentropy from logits hardcoded
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

        return {
            'AE loss min' : tf.reduce_min(ae_multiloss),
            'AE loss max' : tf.reduce_max(ae_multiloss),
            'disc loss min' : tf.reduce_min(disc_losses),
            'disc loss max' : tf.reduce_max(disc_losses),
            'cheat loss min' : tf.reduce_min(cheat_losses),
            'cheat loss max' : tf.reduce_max(cheat_losses)
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


# TODO: early stopping
# TODO: visualization 

# if __name__ == '__main__':
#     model = AAEModel((1234,))
#     model.summary(True)

