import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL

def _compute_number_of_neurons(params,ae):
	neurons = [params['ae_neuron_number_seed']]
	tmp = params['ae_neuron_number_seed']
	for _ in range(params['ae_number_of_layers'] if ae else params['disc_number_of_layers']):
		tmp = int(tmp / 2)
		neurons.append(tmp)
	return neurons


def _build_encoder(molecule_shape,latent_dim,hparams):
	model = keras.models.Sequential(name='Encoder')
	model.add(keras.Input(shape=molecule_shape,name='enc.input'))

	neurons = _compute_number_of_neurons(hparams,ae=True)

	model.add(keras.layers.Dense(neurons[0], activation=hparams['activation'],name='enc.0'))
	model.add(keras.layers.BatchNormalization(momentum=0.8))

	# hidden layers
	for i in range(hparams['ae_number_of_layers']):
		model.add(keras.layers.Dense(neurons[i+1], activation=hparams['activation'], name=f'enc.{i+1}'))
		model.add(keras.layers.BatchNormalization(momentum=0.8))

	#output layer
	model.add(keras.layers.Dense(latent_dim, activation="linear",name='enc.output'))

	return model


def _build_decoder(molecule_shape,latent_dim,hparams):
	model = keras.models.Sequential(name='Decoder')
	model.add(keras.Input(shape=(latent_dim,),name='dec.input'))
	
	neurons = _compute_number_of_neurons(hparams,ae=True)[::-1]
	
	model.add(keras.layers.Dense(neurons[0], activation="linear",name='dec.0'))
	model.add(keras.layers.BatchNormalization(momentum=0.8))
	
	# hidden layers
	for i in range(hparams['ae_number_of_layers']):
		model.add(keras.layers.Dense(neurons[i+1], activation=hparams['activation'],name=f'dec.{i+1}'))
		model.add(keras.layers.BatchNormalization(momentum=0.8))

	# output layer
	model.add(keras.layers.Dense(np.prod(molecule_shape), activation=hparams['activation'],name='dec.output'))
	model.add(keras.layers.Reshape(molecule_shape,name='dec.reshape'))
	return model


def _build_discriminator(latent_dim,hparams):
	model = keras.models.Sequential(name='Discriminator')
	model.add(keras.Input(shape=(latent_dim,),name='disc.input'))

	neurons = _compute_number_of_neurons(hparams,ae=False)

	# model.add(keras.layers.Flatten(input_shape=(latent_dim,)))
	# hidden layers
	for i in range(hparams['disc_number_of_layers']):
		model.add(keras.layers.Dense(neurons[i],name=f'disc.{i}'))
		model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Dense(1,name='disc.output'))

	return model



_default_hp = {
	'batch_size' : 64,
	'activation' : 'relu',
	'ae_number_of_layers': 2,
	'disc_number_of_layers': 2,
	'ae_neuron_number_seed' : 32,
	'disc_neuron_number_seed' : 32,
	'ae_loss_fn': 'MeanSquaredError',
	'disc_loss_fn': 'BinaryCrossentropy',
	'optimizer': 'Adam',
#	'ae_loss_fn': keras.losses.MeanSquaredError(),
#	'disc_loss_fn': keras.losses.BinaryCrossentropy(from_logits=True),
#	'optimizer': keras.optimizers.Adam(0.0002,0.5),
}


_learning_rate = .0002
_optimizers = {
    'Adam':keras.optimizers.legacy.Adam(learning_rate=_learning_rate,beta_1=0.5),
    'SGD':keras.optimizers.legacy.SGD(learning_rate=_learning_rate),
    'RMSprop':keras.optimizers.legacy.RMSprop(learning_rate=_learning_rate),
    'Adadelta':keras.optimizers.legacy.Adadelta(learning_rate=_learning_rate),
    'Adagrad':keras.optimizers.legacy.Adagrad(learning_rate=_learning_rate),
    'Adamax':keras.optimizers.legacy.Adamax(learning_rate=_learning_rate),
    'Nadam':keras.optimizers.legacy.Nadam(learning_rate=_learning_rate),
    'Ftrl':keras.optimizers.legacy.Ftrl(learning_rate=_learning_rate)
}

_losses = {
    'MeanSquaredError' : keras.losses.MeanSquaredError(),
    'Huber' :keras.losses.Huber(),
    'BinaryCrossentropy' :keras.losses.BinaryCrossentropy(from_logits=True),
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


class AAEModel(keras.models.Model):
	def __init__(self,molecule_shape,latent_dim=2,prior='normal',hp=_default_hp):
		super().__init__()
		self.hp = hp
		self.latent_dim = latent_dim
		if prior == 'normal':
			self.get_prior = _PriorNormal(latent_dim)
		elif prior == 'uniform':
			self.get_prior = _PriorUniform(latent_dim)
		else: 
			self.get_prior = _PriorImage(latent_dim,prior)
		

		self.enc = _build_encoder(molecule_shape,latent_dim,hp)
		self.dec = _build_decoder(molecule_shape,latent_dim,hp)
		self.disc = _build_discriminator(latent_dim,hp)

	def compile(self,optimizer=None,ae_loss=None,disc_loss=None): 
		super().compile()
		self.optimizer = _optimizers[optimizer if optimizer else self.hp['optimizer']]
		self.ae_loss_fn = _losses[ae_loss if ae_loss else self.hp['ae_loss_fn']]
		self.disc_loss_fn = _losses[disc_loss if disc_loss else self.hp['disc_loss_fn']]

		self.enc.compile(self.optimizer)
		self.dec.compile(self.optimizer)
		self.disc.compile(self.optimizer)

	
	@tf.function	
	def train_step(self,batch):
		if isinstance(batch,tuple):
			batch = batch[0]

		batch_size = self.hp['batch_size']

# improve AE to reconstruct
		with tf.GradientTape() as ae_tape:
			reconstruct = self.dec(self.enc(batch))
			ae_loss = self.ae_loss_fn(batch,reconstruct)

		enc_dec_weights = self.enc.trainable_weights + self.dec.trainable_weights
		enc_dec_grads = ae_tape.gradient(ae_loss, enc_dec_weights)
		self.optimizer.apply_gradients(zip(enc_dec_grads,enc_dec_weights))
#		grad_vars = self.optimizer.compute_gradients(ae_loss,enc_dec_weights,ae_tape)
#		self.optimizer.apply_gradients(grad_vars)

# improve discriminator
		rand_low = self.get_prior((batch_size,))
		better_low = self.enc(batch)
		low = tf.concat([rand_low,better_low],axis=0)

		labels = tf.concat([tf.ones((batch_size,1)), tf.zeros((batch_size,1))], axis=0)
		labels += 0.05 * tf.random.uniform(tf.shape(labels))	# guide

		with tf.GradientTape() as disc_tape:
			pred = self.disc(low)
			disc_loss = self.disc_loss_fn(labels,pred)

		disc_grads = disc_tape.gradient(disc_loss,self.disc.trainable_weights)
		self.optimizer.apply_gradients(zip(disc_grads,self.disc.trainable_weights))

# teach encoder to cheat
		alltrue = tf.ones((batch_size,1))

		with tf.GradientTape() as cheat_tape:
			cheat = self.disc(self.enc(batch))
			cheat_loss = self.disc_loss_fn(alltrue,cheat)

		cheat_grads = cheat_tape.gradient(cheat_loss,self.enc.trainable_weights)
		self.optimizer.apply_gradients(zip(cheat_grads,self.enc.trainable_weights))

		return { 'ae_loss' : ae_loss, 'd_loss' : disc_loss }

	def summary(self,expand_nested=True):
		self.enc.summary(expand_nested=expand_nested)
		self.dec.summary(expand_nested=expand_nested)
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
	

# TODO: image prior
# TODO: early stopping
# TODO: visualization 

if __name__ == '__main__':
	model = AAEModel((1234,))
	model.summary(True)
