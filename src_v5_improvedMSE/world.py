'''
Created on June 12, 2025
Updated world configuration with MSE loss and advanced training parameters
Optimized for spectral collaborative filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import os
import torch
from parse import parse_args, validate_args
import multiprocessing

# Parse and validate arguments
args = parse_args()
args = validate_args(args)

config = {}

# Basic configuration (use args after potential preset modifications)
config['train_u_batch_size'] = args.train_u_batch_size
config['eval_u_batch_size'] = args.eval_u_batch_size
config['dataset'] = args.dataset
config['lr'] = args.lr
config['decay'] = args.decay
config['epochs'] = args.epochs
config['filter'] = args.filter
config['filter_order'] = args.filter_order
config['verbose'] = args.verbose
config['patience'] = args.patience
config['min_delta'] = args.min_delta
config['n_epoch_eval'] = args.n_epoch_eval

# MSE Loss parameters (optimized for matrix-based models)
config['samples'] = getattr(args, 'samples', 50)  # Number of negative samples
config['pos_weight'] = getattr(args, 'pos_weight', 1.0)  # Positive samples weight
config['neg_weight'] = getattr(args, 'neg_weight', 0.5)  # Negative samples weight
config['hard_neg_ratio'] = getattr(args, 'hard_neg_ratio', 0.3)  # Hard negatives ratio

# Embedding configuration
config['embed_dim'] = getattr(args, 'embed_dim', 64)  # Embedding dimension

# Eigenvalue configuration (ensure they get passed correctly)
config['u_n_eigen'] = args.u_n_eigen if hasattr(args, 'u_n_eigen') else None
config['i_n_eigen'] = args.i_n_eigen if hasattr(args, 'i_n_eigen') else None
config['b_n_eigen'] = args.b_n_eigen if hasattr(args, 'b_n_eigen') else None

# Filter configurations (simplified)
config['user_filter_design'] = getattr(args, 'user_filter_design', 'simple')
config['item_filter_design'] = getattr(args, 'item_filter_design', 'simple')
config['bipartite_filter_design'] = getattr(args, 'bipartite_filter_design', 'simple')
config['user_init_filter'] = args.user_init_filter
config['item_init_filter'] = args.item_init_filter
config['bipartite_init_filter'] = args.bipartite_init_filter

# System configuration
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
config['device'] = device

CORES = multiprocessing.cpu_count() // 2
seed = args.seed
dataset = args.dataset
model_name = args.model
TRAIN_epochs = args.epochs
topks = eval(args.topks)

def cprint(words: str):
    """Colored print for dataset loading"""
    print(f"\033[0;30;43m{words}\033[0m")

def print_config_summary():
    """Print a nice configuration summary"""
    if config['verbose'] > 0:
        print(f"\n🎛️  \033[96mImproved Universal Spectral CF Configuration (MSE Loss)\033[0m")
        print(f"┌─ Dataset: \033[94m{dataset}\033[0m")
        print(f"├─ Device: \033[94m{device}\033[0m")
        print(f"├─ Filter combo: \033[94m{config['filter']}\033[0m")
        print(f"├─ Filter order: \033[94m{config['filter_order']}\033[0m")
        print(f"├─ Embedding dim: \033[94m{config['embed_dim']}\033[0m")
        print(f"├─ Learning rate: \033[94m{config['lr']}\033[0m")
        print(f"├─ Epochs: \033[94m{config['epochs']}\033[0m")
        print(f"├─ Negative samples: \033[94m{config['samples']}\033[0m")
        print(f"├─ Positive weight: \033[94m{config['pos_weight']}\033[0m")
        print(f"├─ Negative weight: \033[94m{config['neg_weight']}\033[0m")
        print(f"├─ Hard neg ratio: \033[94m{config['hard_neg_ratio']}\033[0m")
        print(f"└─ Seed: \033[94m{seed}\033[0m")

# Available filter types for reference (simplified)
AVAILABLE_FILTERS = {
    'simple': 'Simple learnable spectral filter',
    'original': 'Original universal spectral filter',
    'enhanced_basis': 'Multi-pattern enhanced filter',
    'chebyshev': 'Chebyshev polynomial filter'
}

AVAILABLE_PATTERNS = {
    'smooth': 'Smooth low-pass pattern',
    'sharp': 'Sharp edge-preserving pattern',
    'bandpass': 'Band-pass frequency pattern',
    'golden_036': 'Golden ratio optimized pattern'
}

def get_optimized_mse_config():
    """Get optimized configuration for MSE training"""
    return {
        'lr': 0.001,
        'decay': 1e-4,
        'samples': 50,  # Multiple negative samples
        'pos_weight': 1.0,  # Standard positive weight
        'neg_weight': 0.5,  # Reduced negative weight
        'hard_neg_ratio': 0.3,  # Hard negative mining
        'embed_dim': 64,
        'filter_order': 6,
        'epochs': 150,
        'train_u_batch_size': 1024,
        'eval_u_batch_size': 500,
        'patience': 10,
        'n_epoch_eval': 5
    }

def get_ml100k_mse_config():
    """Optimized MSE config for ML-100K dataset"""
    return {
        'lr': 0.001,
        'decay': 1e-4,
        'samples': 30,
        'pos_weight': 1.0,
        'neg_weight': 0.3,
        'hard_neg_ratio': 0.2,
        'embed_dim': 64,
        'filter': 'ui',
        'epochs': 150
    }

def get_gowalla_mse_config():
    """Optimized MSE config for Gowalla dataset"""
    return {
        'lr': 0.01,
        'decay': 1e-3,
        'samples': 100,
        'pos_weight': 1.0,
        'neg_weight': 0.4,
        'hard_neg_ratio': 0.3,
        'embed_dim': 64,
        'filter': 'uib',
        'epochs': 50
    }

# Print configuration summary at startup
if __name__ != '__main__':
    print_config_summary()