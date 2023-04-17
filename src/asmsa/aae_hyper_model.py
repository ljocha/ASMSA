#! vim: ai ts=4:
import keras_tuner as kt
import numpy as np
from tensorflow import keras
import tensorflow as tf

from .aae_model import AAEModel



def _KLdivergence(x, y):
	"""Compute the Kullback-Leibler divergence between two multivariate samples.
	Parameters
	----------
	x : 2D array (n,d)
	Samples from distribution P, which typically represents the true
	distribution.
	y : 2D array (m,d)
	Samples from distribution Q, which typically represents the approximate
	distribution.
	Returns
	-------
	out : float
	The estimated Kullback-Leibler divergence D(P||Q).
	References
	----------
	Pérez-Cruz, F. Kullback-Leibler divergence estimation of
	continuous distributions IEEE International Symposium on Information
	Theory, 2008.
	"""
	from scipy.spatial import cKDTree as KDTree

	# Check the dimensions are consistent
	x = np.atleast_2d(x)
	y = np.atleast_2d(y)

	n,d = x.shape
	m,dy = y.shape

	assert(d == dy)


	# Build a KD tree representation of the samples and find the nearest neighbour
	# of each point in x.
	xtree = KDTree(x)
	ytree = KDTree(y)

	# Get the first two nearest neighbours for x, since the closest one is the
	# sample itself.
	r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
	s = ytree.query(x, k=1, eps=.01, p=2)[0]

	# There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
	# on the first term of the right hand side.
	return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

class _KLGatherCallback(keras.callbacks.Callback):
	def __init__(self,model,valdata):
		super().__init__()
		self.set_model(model)
		self.valdata = valdata

	def on_train_begin(self,logs=None):
		self.kl = []

	def on_epoch_end(self,epoch,logs=None):
		print('\n\nPosterior on valdata ...',end='')
		prior = self.model.get_prior((self.valdata.shape[0],)).numpy()
		posterior = self.model.call_enc(self.valdata).numpy()

		kls = []
		print('\nKL divergences ',end='')
		for m in range(self.model.n_models):
			kl = _KLdivergence(prior,posterior[:,m*self.model.latent_dim:(m+1)*self.model.latent_dim])
			if kl < 0.: kl = 0. 
			kls.append(kl)
			print('.',end='')
		print()
		
		self.kl.append(kls)

	def get_metric(self,start=0.):
		nkl = np.array(self.kl[int(len(self.kl) * start):])
		klmean = np.mean(nkl,axis=0)
		klvar = np.var(nkl,axis=0)

		return np.min(klmean + klvar)

		

class AAEHyperModel(kt.HyperModel):

	def __init__(self,molecule_shape,latent_dim=2,tuning_threshold=.25,prior='normal',hp=None):
		super().__init__()
		assert hp
		self.keras_hp = {}

		for k,v in hp.items():
			if isinstance(v,range):
				v = list(v)
			elif not isinstance(v,list):
				v = [ v ]

			if k == 'ae_neuron_number_seed':
				aes = v
			elif k == 'disc_neuron_number_seed':
				dis = v
			else:
				self.keras_hp[k] = v

		el = len(aes)
		dl = len(dis)
		self.enc_seeds = aes * dl
		self.disc_seeds = list(np.repeat(np.array(dis),el,axis=0))
		
		self.molecule_shape = molecule_shape
		self.latent_dim = latent_dim
		self.prior = prior
		self.tuning_threshold = tuning_threshold


	def build(self,hp):
		for k,v in self.keras_hp.items():
			hp.Choice(k,v)

		mod = AAEModel(molecule_shape=self.molecule_shape,latent_dim=self.latent_dim,prior=self.prior,
			enc_layers = hp['ae_number_of_layers'], enc_seed = self.enc_seeds,
			disc_layers = hp['disc_number_of_layers'], disc_seed = self.disc_seeds,
			hp=hp)
		mod.compile()
		return mod


	def fit(self, hp, model, train, validation=None, callbacks=[], **kwargs):
		if validation is None:
			validation = train[::5]		# XXX 
		klcb = _KLGatherCallback(model,validation)
		train = tf.data.Dataset.from_tensor_slices(train)\
			.cache()\
			.shuffle(2048)\
			.batch(hp['batch_size'],drop_remainder=True)\
			.prefetch(tf.data.experimental.AUTOTUNE)

		super().fit(hp, model, train, callbacks=callbacks + [klcb], **kwargs)

		return klcb.get_metric(self.tuning_threshold)

