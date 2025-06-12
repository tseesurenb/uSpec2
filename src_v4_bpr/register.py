'''
Created on June 12, 2025
Updated model registration for improved spectral CF
Uses the enhanced model with BPR loss and embeddings

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''
import world
import dataloader

# Dataset loading
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'ml-100k':
    dataset = dataloader.ML100K()
else:
    raise ValueError(f"Unknown dataset: {world.dataset}")

# Model selection - Updated to use improved model
from model import ImprovedUniversalSpectralCF
MODELS = {'uspec': ImprovedUniversalSpectralCF}