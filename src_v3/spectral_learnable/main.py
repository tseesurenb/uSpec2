'''
Created on June 12, 2025
Simplified Universal Spectral CF main file
Minimalist approach similar to GF-CF

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import world
import utils
import procedure
import time
from register import dataset, MODELS
import warnings
warnings.filterwarnings("ignore")

# Set random seed
utils.set_seed(world.seed)

print(f"Universal Spectral CF: {world.config['dataset']} dataset")
print(f"Filter: {world.config['filter']}, Device: {world.device}")

# Create model
print(f"Creating model...")
model_start = time.time()
adj_mat = dataset.UserItemNet.tolil()

# Choose model based on config
model_name = world.config.get('model', 'uspec')
if model_name == 'gfcf':
    from gf_cf_model import GF_CF_Model
    Recmodel = GF_CF_Model(adj_mat, world.config)
else:
    UniversalSpectralCF = MODELS['uspec']
    Recmodel = UniversalSpectralCF(adj_mat, world.config)
print(f"Model created in {time.time() - model_start:.2f}s")

# Dataset info
print(f"Dataset: {dataset.n_users:,} users, {dataset.m_items:,} items")
print(f"Training: {dataset.trainDataSize:,} interactions")

# Training - procedure.py handles all printing
trained_model, final_results = procedure.train_and_evaluate(dataset, Recmodel, world.config)