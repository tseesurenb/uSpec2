'''
Created on June 12, 2025
Simplified model registration
Minimalist approach

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''
import world
import dataloader

# Dataset loading
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'ml-100k':
    dataset = dataloader.ML100K()
else:
    raise ValueError(f"Unknown dataset: {world.dataset}")

# Model selection - Fixed import
import model
from model_simplified import SimplifiedSpectralCF

# Use simplified model by default (more like static model)
MODELS = {
    'uspec': SimplifiedSpectralCF,  # Default to simplified version
    'uspec_complex': model.UserSpecificUniversalSpectralCF  # Original complex version
}
