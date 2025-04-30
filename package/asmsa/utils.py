import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL
import tensorflow_probability as tfp
import math

def _compute_number_of_neurons(layers, seed):
    neurons = [seed]
    tmp = seed
    for _ in range(layers):
        tmp = int(tmp / 2)
        neurons.append(tmp)
    return neurons

_default_hp = {
    'activation': 'gelu',
    'ae_loss_fn': 'MeanSquaredError',
    'optimizer': 'Adam',
    'learning_rate': 0.0002
}

_losses = {
    'MeanSquaredError': keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE),
    'Huber': keras.losses.Huber(reduction=keras.losses.Reduction.NONE),
}

class _Prior():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def __call__(self, shape):
        pass

class _PriorDistribution(_Prior):
    def __init__(self, latent_dim, prior):
        super().__init__(latent_dim)
        self.prior = prior

    @tf.function
    def __call__(self, shape):
        sample = tfp.distributions.Sample(
            self.prior,
            sample_shape=(*shape, self.latent_dim)).sample()

        if len(sample.shape) > 2:
            sample = tf.reshape(sample, [sample.shape[0] * sample.shape[1], self.latent_dim])

        return sample

class _PriorMultivariate(_Prior):
    def __init__(self, prior):
        super().__init__(prior.event_shape[0])
        self.prior = prior

    @tf.function
    def __call__(self, shape):
        return self.prior.sample(shape)

class _PriorImage(_Prior):
    def __init__(self, latent_dim, file):
        super().__init__(latent_dim)
        img = np.array(PIL.Image.open(file), dtype=np.float64)
        if len(img.shape) == 3:
            img = np.sum(img, axis=2)

        assert len(img.shape) == 2 and latent_dim == 2

        img /= np.sum(img)
        self.shape = img.shape
        self.flat = tf.convert_to_tensor(np.cumsum(img.flatten()).astype(np.float32))

    @tf.function
    def __call__(self, shape):
        u = tf.random.uniform(shape=shape)
        b = tf.broadcast_to(self.flat, u.shape[:-1] + self.flat.shape)
        flati = tf.searchsorted(b, u)
        x = tf.cast(flati % self.shape[1], tf.float32)
        x += tf.random.uniform(x.shape)
        y = tf.cast(flati // self.shape[1], tf.float32)
        y += tf.random.uniform(y.shape)
        x /= self.shape[1]
        y = 1. - y / self.shape[0]
        return tf.stack([x, y], axis=-1)

_random_init = keras.initializers.GlorotUniform(seed=42)

class _SparseConstraint(keras.constraints.Constraint):
    def __init__(self, left, right):
        super().__init__()
        assert len(left) == len(right)
        mask = np.zeros((np.sum(left), np.sum(right)), dtype=np.float32)
        idxl = np.concatenate((np.zeros((1,), dtype=np.int32), np.cumsum(left)))
        idxr = np.concatenate((np.zeros((1,), dtype=np.int32), np.cumsum(right)))
        for mod in range(len(left)):
            mask[idxl[mod]:idxl[mod + 1], idxr[mod]:idxr[mod + 1]] = 1.
        self.mask = tf.convert_to_tensor(mask)

    def __call__(self, w):
        return w * self.mask

class _SparseInitializer(keras.initializers.Initializer):
    def __init__(self, left, right):
        super().__init__()
        assert len(left) == len(right)
        self.left = left
        self.right = right
        self.idxl = np.concatenate((np.zeros((1,), dtype=np.int32), np.cumsum(left)))
        self.idxr = np.concatenate((np.zeros((1,), dtype=np.int32), np.cumsum(right)))

    def __call__(self, shape, dtype=None):
        assert list(shape) == [np.sum(self.left), np.sum(self.right)]
        init = np.zeros((np.sum(self.left), np.sum(self.right)), dtype=dtype)
        for mod in range(len(self.left)):
            init[self.idxl[mod]:self.idxl[mod + 1], self.idxr[mod]:self.idxr[mod + 1]] = _random_init(
                (self.left[mod], self.right[mod])).numpy()
        return tf.convert_to_tensor(init)

def _masks(left, right):
    return {'kernel_initializer': _SparseInitializer(left, right),
            'kernel_constraint': _SparseConstraint(left, right)}
