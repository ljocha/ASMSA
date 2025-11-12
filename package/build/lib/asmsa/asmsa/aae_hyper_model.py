#! vim: ai ts=4:
import keras_tuner as kt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
from datetime import datetime
import dict_hash 
from pathlib import Path
import os
import copy

from .aae_model import AAEModel
from torch.utils.tensorboard import SummaryWriter



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


class _LogCallback(keras.callbacks.Callback):
    def __init__(self,model,valdata,tuning_threshold):
        super().__init__()
        self.set_model(model)
        self.valdata = valdata
        # make trial_id unique throughout tunings
        hp_values = copy.deepcopy(self.model.hp.values)
        hp_values['datettime'] = datetime.today().strftime("%m%d%Y-%H%M%S.%f")
        self.trial_id = dict_hash.sha256(hp_values)
        self.tuning_threshold = tuning_threshold

    def on_train_begin(self,logs=None):
        self.kl = []
        print(f'Trial ID: {self.trial_id}')

        # set template for writing of results
        self.d = {
            "trial_id" : self.trial_id,
            "hp" : self.model.hp.values,
            "results" : {},
            "score" : None
        }
        
        self.multibatch = tf.stack([self.valdata]*self.model.n_models,axis=1)
        
    def on_train_end(self,epoch,logs=None):
        self.d['score'] = self.get_metric(self.tuning_threshold)
        workdir = os.getcwd()
        tuning_path = f'{workdir}/analysis/{os.environ["RESULTS_DIR"]}'
        
        Path(tuning_path).mkdir(parents=True, exist_ok=True)
        obj = f"{tuning_path}/{self.trial_id}"
        pickle.dump( self.d, open( obj, "wb" ) )

    def on_epoch_end(self,epoch,logs=None):
        prior = self.model.get_prior((self.valdata.shape[0],)).numpy()
        posterior = self.model.call_enc(self.valdata).numpy()

        kls = []
        for m in range(self.model.n_models):
            kl = _KLdivergence(prior,posterior[:,m*self.model.latent_dim:(m+1)*self.model.latent_dim])
            if kl < 0.: kl = 0. 
            kls.append(kl)
            print('.',end='')
        print()
        
        self.kl.append(kls)
        
        # compute AE loss
        reconstruct = self.model.aes(self.valdata)
        mse = self.model.ae_loss_fn(self.multibatch,reconstruct[0])
        ae_multiloss = tf.reduce_mean(mse,axis=0)

        rand_low = self.model.get_prior((self.valdata.shape[0],))
        rand_low = tf.tile(rand_low,(1,self.model.n_models))

        # compute discriminator loss
        neg_pred = self.model.disc(reconstruct[1])
        neg_losses = tf.reduce_mean(neg_pred*tf.random.uniform(tf.shape(neg_pred),1.,1.05),axis=0) 
        pos_pred = self.model.disc(rand_low)
        pos_losses = -tf.reduce_mean(pos_pred*tf.random.uniform(tf.shape(pos_pred),1.,1.05),axis=0)
        dn_losses = neg_losses + pos_losses

        for i in range(0, len(self.model.enc_seed)):
            model = f"model_AE{self.model.enc_seed[i]}_DN{self.model.disc_seed[i]}"
            
            self._log(model=model,
                      ae=ae_multiloss[i].numpy(),
                      dn=dn_losses[i].numpy(),
                      kl=kls[i])
            
    def _log(self,model,ae=None,dn=None,kl=None):
        try:
            self.d['results'][model]['ae_loss'].append(ae)
            self.d['results'][model]['dn_loss'].append(dn)
            self.d['results'][model]['kl_div'].append(kl)
        except KeyError:
            self.d['results'][model] = {
                'ae_loss' : [],
                'dn_loss' : [],
                'kl_div' : []
            }
            self._log(model,ae,dn,kl)
        

    def get_metric(self,start=0.):
        nkl = np.array(self.kl[int(len(self.kl) * start):])
        klmean = np.mean(nkl,axis=0)
        klvar = np.var(nkl,axis=0)

        return np.min(klmean + klvar)


class AAEHyperModel(kt.HyperModel):

    def __init__(self,molecule_shape,latent_dim=2,tuning_threshold=.25,prior=tfp.distributions.Normal(loc=0, scale=1),hp=None):
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
        logcb = _LogCallback(model,validation,self.tuning_threshold)
        train = tf.data.Dataset.from_tensor_slices(train)\
            .cache()\
            .shuffle(2048)\
            .batch(hp['batch_size'],drop_remainder=True)\
            .prefetch(tf.data.experimental.AUTOTUNE)

        super().fit(hp, model, train, callbacks=callbacks + [logcb], **kwargs)

        return logcb.get_metric(self.tuning_threshold)

