#!/usr/bin/env python3

import tensorflow as tf
import os
threads=int(os.environ['OMP_NUM_THREADS'])
tf.config.threading.set_inter_op_parallelism_threads(threads)
tf.config.threading.set_intra_op_parallelism_threads(threads)

import argparse
import mdtraj as md
import numpy as np
from tensorflow import keras
import keras_tuner
import asmsa
import dill

p = argparse.ArgumentParser()
p.add_argument('--pdb')
p.add_argument('--xtc')
p.add_argument('--top')
p.add_argument('--ndx')
p.add_argument('--output')
p.add_argument('--trials')
p.add_argument('--epochs')
p.add_argument('--hpfunc')
p.add_argument('--id')
p.add_argument('--master')

a = p.parse_args()

conf = a.pdb
traj = a.xtc
topol = a.top
index = a.ndx
output = a.output
trials = int(a.trials) if a.trials else 42
epochs = int(a.epochs) if a.epochs else 7
tuner_id = a.id if a.id else os.environ['HOSTNAME']

assert (tuner_id != 'chief' or output)

with open(a.hpfunc,'rb') as p:
	hpfunc = dill.load(p)

os.environ['KERASTUNER_TUNER_ID'] = tuner_id
os.environ['KERASTUNER_ORACLE_IP'] = a.master.split(':')[0]
os.environ['KERASTUNER_ORACLE_PORT'] = a.master.split(':')[1]

tr = md.load(traj,top=conf)
idx=tr[0].top.select("name CA")
tr.superpose(tr[0],atom_indices=idx)
geom = np.moveaxis(tr.xyz ,0,-1)

density = 2 # integer in [1, n_atoms-1]
sparse_dists = asmsa.NBDistancesSparse(geom.shape[0], density=density)
mol = asmsa.Molecule(pdb=conf,top=topol,ndx=index,fms=[sparse_dists])

X_train = mol.intcoord(geom).T

def full_hp(hp):
    return {
        'activation' : hp.Choice('activation', ['relu', 'gelu', 'selu']),    
        'ae_neuron_number_seed' : hp.Int("ae_neuron_number_seed", 32, 224, step=64),
        'disc_neuron_number_seed' : hp.Int("disc_neuron_number_seed", 32, 224, step=64),
        'ae_number_of_layers' : hp.Int("ae_number_of_layers", 2, 3, step=1),
        'disc_number_of_layers' : hp.Int("disc_number_of_layers", 2, 3, step=1),
        'batch_size' : hp.Int("batch_size", 32, 256, step=64),
        'optimizer' : hp.Choice('optimizer', ['Adam', 'SGD', 'RMSProp', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam','Ftrl' ]),
        'ae_loss_fn' : hp.Choice('ae_loss_fn', [ 'MeanSquaredError', 'Huber']),
        'disc_loss_fn' : hp.Choice('disc_loss_fn', ['BinaryCrossentropy'])
    }

def medium_hp(hp):
    return {
        'activation' : hp.Choice('activation', ['relu', 'gelu']),    
        'ae_neuron_number_seed' : hp.Int("ae_neuron_number_seed", 32, 224, step=64),
        'disc_neuron_number_seed' : hp.Int("disc_neuron_number_seed", 32, 224, step=64),
        'ae_number_of_layers' : hp.Int("ae_number_of_layers", 2, 2),
        'disc_number_of_layers' : hp.Int("disc_number_of_layers", 2, 2),
        'batch_size' : hp.Int("batch_size", 64, 128, step=64),
        'optimizer' : hp.Choice('optimizer', ['Adam']),
        'ae_loss_fn' : hp.Choice('ae_loss_fn', ['MeanSquaredError']),
        'disc_loss_fn' : hp.Choice('disc_loss_fn', ['BinaryCrossentropy'])  
    }

def tiny_hp(hp):
    return {
        'activation' : hp.Choice('activation', ['relu']),    
        'ae_neuron_number_seed' : hp.Int("ae_neuron_number_seed", 32, 64, step=32),
        'disc_neuron_number_seed' : hp.Int("disc_neuron_number_seed", 32, 32),
        'ae_number_of_layers' : hp.Int("ae_number_of_layers", 2, 2),
        'disc_number_of_layers' : hp.Int("disc_number_of_layers", 2, 2),
        'batch_size' : hp.Int("batch_size", 64, 128, step=64),
        'optimizer' : hp.Choice('optimizer', ['Adam']),
        'ae_loss_fn' : hp.Choice('ae_loss_fn', ['MeanSquaredError']),
        'disc_loss_fn' : hp.Choice('disc_loss_fn', ['BinaryCrossentropy'])  
    }



tuner = keras_tuner.RandomSearch(
	max_trials=trials,
	hypermodel=asmsa.AAEHyperModel((X_train.shape[1],),hpfunc=hpfunc),
	objective=keras_tuner.Objective("score", direction="min"),
	directory="./results",
	project_name=tuner_id,
	overwrite=True
)

tuner.search(X_train,epochs=epochs)

print(f"{tuner_id}: Done!")
