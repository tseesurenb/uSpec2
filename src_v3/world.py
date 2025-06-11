'''
Created on June 7, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Enhanced with user-specific model configuration support and BCE loss

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import os
import torch
from parse import parse_args
import multiprocessing

args = parse_args()

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'ml-100k']
all_models = ['uspec']

# Basic training parameters
config['train_u_batch_size'] = args.train_u_batch_size
config['eval_u_batch_size'] = args.eval_u_batch_size
config['dataset'] = args.dataset
config['lr'] = args.lr
config['decay'] = args.decay
config['epochs'] = args.epochs
config['filter'] = args.filter
config['filter_order'] = args.filter_order
config['verbose'] = args.verbose
config['val_ratio'] = args.val_ratio
config['patience'] = args.patience
config['min_delta'] = args.min_delta
config['n_epoch_eval'] = args.n_epoch_eval
config['m_type'] = args.m_type

# Loss function configuration (NEW!)
config['loss_function'] = args.loss_function

# BCE-specific parameters (NEW!)
config['bce_pos_weight'] = args.bce_pos_weight
config['negative_sampling_ratio'] = args.negative_sampling_ratio
config['use_focal_loss'] = args.use_focal_loss
config['focal_alpha'] = args.focal_alpha
config['focal_gamma'] = args.focal_gamma

# Model selection - UPDATED WITH USER-SPECIFIC MODEL
config['model_type'] = args.model_type

# Enhanced eigenvalue configuration with three-view support
config['n_eigen'] = args.n_eigen  # Legacy support
config['u_n_eigen'] = args.u_n_eigen  # User eigenvalue count
config['i_n_eigen'] = args.i_n_eigen  # Item eigenvalue count
config['b_n_eigen'] = args.b_n_eigen  # Bipartite eigenvalue count

# Filter design options (only used with enhanced model)
config['filter_design'] = args.filter_design
config['init_filter'] = args.init_filter

# Simple model specific parameters
config['filter_mode'] = args.filter_mode
config['n_hops'] = args.n_hops

# User-specific model parameters (NEW!)
config['shared_base'] = args.shared_base
config['personalization_dim'] = args.personalization_dim
config['cold_start_strategy'] = args.cold_start_strategy

# Laplacian-specific configuration (legacy)
config['use_laplacian'] = args.use_laplacian
config['laplacian_type'] = args.laplacian_type

# Enhanced similarity-aware configuration (only used with enhanced model)
config['use_similarity_norm'] = args.use_similarity_norm
config['similarity_type'] = args.similarity_type
config['similarity_threshold'] = args.similarity_threshold
config['similarity_weight'] = args.similarity_weight

# Polynomial filter parameters
config['polynomial_type'] = args.polynomial_type
config['polynomial_params'] = {
    'alpha': args.polynomial_alpha,
    'beta': args.polynomial_beta,
    'types': args.adaptive_polynomial_types
}

config['n_hops'] = args.n_hops
config['hop_decay'] = args.hop_decay

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
config['device'] = device  # Add device to config for model access

CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

TRAIN_epochs = args.epochs
topks = eval(args.topks)

def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")