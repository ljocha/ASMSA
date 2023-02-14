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
		prior = self.model.get_prior((self.valdata.shape[0],))
		posterior = self.model.call_enc(self.valdata)
		kl = _KLdivergence(prior,posterior)
		if kl < 0.: kl = 0. 
		
		self.kl.append(kl)

	def get_metric(self,start=0.):
		nkl = np.array(self.kl[int(len(self.kl) * start):])
		return [np.var(nkl), np.mean(nkl)]

		

class AAEHyperModel(kt.HyperModel):

	def __init__(self,molecule_shape,latent_dim=2,tuning_threshold=.25,prior='normal',hpfunc=None):
		super().__init__()
		assert hpfunc
		self.hpfunc = hpfunc
		self.molecule_shape = molecule_shape
		self.latent_dim = latent_dim
		self.prior = prior
		self.tuning_threshold = tuning_threshold


	def build(self,hp):
		myhp = self.hpfunc(hp)
		mod = AAEModel(molecule_shape=self.molecule_shape,latent_dim=self.latent_dim,prior=self.prior,hp=myhp)
		mod.compile()
		return mod


	def fit(self, hp, model, x, callbacks=[], **kwargs):
		klcb = _KLGatherCallback(model,x)
		train = tf.data.Dataset.from_tensor_slices(x)\
			.cache()\
			.shuffle(2048)\
			.batch(hp['batch_size'],drop_remainder=True)\
			.prefetch(tf.data.experimental.AUTOTUNE)

		super().fit(hp, model, train, callbacks=callbacks + [klcb], **kwargs)

		return klcb.get_metric(self.tuning_threshold)

