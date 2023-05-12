#! vim: ai ts=4:
import keras_tuner as kt
import numpy as np
from tensorflow import keras
import tensorflow as tf
from datetime import datetime

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

from datetime import datetime
class _KLGatherCallback(keras.callbacks.Callback):
    def __init__(self,model,valdata,trial_id,start_time):
        super().__init__()
        self.set_model(model)
        self.valdata = valdata
        self.trial_id = trial_id
        self.start_time = start_time

    def on_train_begin(self,logs=None):
        self.kl = []
        
        self.summary_writer = SummaryWriter(f'/home/jovyan/ASMSA/multipleruns/{self.start_time}/{self.trial_id}')
        self.multibatch = tf.stack([self.valdata]*self.model.n_models,axis=1)
        
    def on_train_end(self,epoch,logs=None):
        self.summary_writer.close()

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
        
        # compute AE loss
        reconstruct = self.model.aes(self.valdata)
        mse = self.model.ae_loss_fn(self.multibatch,reconstruct[0])
        ae_multiloss = tf.reduce_mean(mse,axis=0)

        rand_low = self.model.get_prior((tf.shape(self.valdata)[0],))
        rand_low = tf.tile(rand_low,(1,self.model.n_models))

        # compute discriminator loss
        neg_pred = self.model.disc(reconstruct[1])
        neg_losses = tf.reduce_mean(neg_pred*tf.random.uniform(tf.shape(neg_pred),1.,1.05),axis=0) 
        pos_pred = self.model.disc(rand_low)
        pos_losses = -tf.reduce_mean(pos_pred*tf.random.uniform(tf.shape(pos_pred),1.,1.05),axis=0)
        dn_losses = neg_losses + pos_losses

        ae_losses = {}
        disc_losses = {}
        kl_divergencies = {}
        for i in range(0, len(self.model.enc_seed)):
            name = f"model_AE{self.model.enc_seed[i]}_DN{self.model.disc_seed[i]}"
            ae_losses[name] = ae_multiloss[i].numpy()
            disc_losses[name] = dn_losses[i].numpy()
            kl_divergencies[name] = kls[i]

        self.summary_writer.add_scalars(f'Autoencoder_loss', ae_losses, epoch)
        self.summary_writer.add_scalars(f'Discriminator_loss', disc_losses, epoch)
        self.summary_writer.add_scalars(f'KL_divergence', kl_divergencies, epoch)
        

    def get_metric(self,start=0.):
        nkl = np.array(self.kl[int(len(self.kl) * start):])
        klmean = np.mean(nkl,axis=0)
        klvar = np.var(nkl,axis=0)

        return np.min(klmean + klvar)


class AAEHyperModel(kt.HyperModel):

    def __init__(self,molecule_shape,latent_dim=2,tuning_threshold=.25,prior='normal',hpfunc=None):
        super().__init__()
        hp = hpfunc
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
        self.trial_id = 1
        self.start_time = datetime.today().strftime("%H-%M-%S")


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
        klcb = _KLGatherCallback(model,validation,self.trial_id,self.start_time)
        # logcb = LogCallback(model,1,validation)
        train = tf.data.Dataset.from_tensor_slices(train)\
            .cache()\
            .shuffle(2048)\
            .batch(hp['batch_size'],drop_remainder=True)\
            .prefetch(tf.data.experimental.AUTOTUNE)

        super().fit(hp, model, train, callbacks=callbacks + [klcb], **kwargs)
        self.trial_id += 1

        return klcb.get_metric(self.tuning_threshold)

