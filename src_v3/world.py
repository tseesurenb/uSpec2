'''
Created on June 12, 2025
Enhanced world configuration with all filter options
Minimalist approach with comprehensive filter support

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

# Basic configuration
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

# Eigenvalue configuration
config['u_n_eigen'] = args.u_n_eigen
config['i_n_eigen'] = args.i_n_eigen
config['b_n_eigen'] = args.b_n_eigen

# Enhanced filter configurations
config['user_filter_design'] = args.user_filter_design
config['item_filter_design'] = args.item_filter_design
config['bipartite_filter_design'] = args.bipartite_filter_design
config['user_init_filter'] = args.user_init_filter
config['item_init_filter'] = args.item_init_filter
config['bipartite_init_filter'] = args.bipartite_init_filter

# Personalization dimensions
config['user_personalization_dim'] = args.user_personalization_dim
config['item_personalization_dim'] = args.item_personalization_dim
config['bipartite_personalization_dim'] = args.bipartite_personalization_dim

# Advanced filter parameters
config['n_bands'] = args.n_bands
config['n_harmonics'] = args.n_harmonics
config['n_stop_bands'] = args.n_stop_bands
config['ensemble_temperature'] = args.ensemble_temperature

# Polynomial filter parameters
config['polynomial_type'] = args.polynomial_type
config['alpha'] = args.alpha
config['beta'] = args.beta

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
        print(f"\n🎛️  \033[96mUniversal Spectral CF Configuration\033[0m")
        print(f"┌─ Dataset: \033[94m{dataset}\033[0m")
        print(f"├─ Device: \033[94m{device}\033[0m")
        print(f"├─ Filter combo: \033[94m{config['filter']}\033[0m")
        print(f"├─ Filter order: \033[94m{config['filter_order']}\033[0m")
        print(f"├─ Learning rate: \033[94m{config['lr']}\033[0m")
        print(f"├─ Epochs: \033[94m{config['epochs']}\033[0m")
        print(f"└─ Seed: \033[94m{seed}\033[0m")
        
        print(f"\n🔧 \033[95mFilter Designs:\033[0m")
        print(f"├─ User: \033[93m{config['user_filter_design']}\033[0m (\033[92m{config['user_init_filter']}\033[0m)")
        print(f"├─ Item: \033[93m{config['item_filter_design']}\033[0m (\033[92m{config['item_init_filter']}\033[0m)")
        print(f"└─ Bipartite: \033[93m{config['bipartite_filter_design']}\033[0m (\033[92m{config['bipartite_init_filter']}\033[0m)")

# Available filter types for reference
AVAILABLE_FILTERS = {
    'original': 'Basic universal spectral filter',
    'spectral_basis': 'Spectral basis filter',
    'enhanced_basis': 'Multi-pattern enhanced filter',
    'chebyshev': 'Chebyshev polynomial filter',
    'jacobi': 'Jacobi polynomial filter',
    'legendre': 'Legendre polynomial filter',
    'laguerre': 'Laguerre polynomial filter',
    'hermite': 'Hermite polynomial filter',
    'bernstein': 'Bernstein polynomial filter',
    'universal_polynomial': 'Universal polynomial filter',
    'bandstop': 'Band-stop frequency filter',
    'adaptive_bandstop': 'Advanced multi-band stop filter',
    'parametric': 'Parametric multi-band filter',
    'multiscale': 'Multi-scale spectral filter',
    'harmonic': 'Harmonic series filter',
    'golden': 'Adaptive golden ratio filter',
    'ensemble': 'Ensemble of all filter types'
}

AVAILABLE_PATTERNS = {
    'smooth': 'Smooth low-pass pattern',
    'sharp': 'Sharp edge-preserving pattern',
    'bandpass': 'Band-pass frequency pattern',
    'golden_036': 'Golden ratio optimized pattern',
    'butterworth': 'Butterworth filter pattern',
    'gaussian': 'Gaussian smoothing pattern',
    'band_stop': 'Band-stop rejection pattern',
    'notch': 'Notch filter pattern'
}

def get_filter_info():
    """Get information about available filters"""
    return {
        'filter_types': AVAILABLE_FILTERS,
        'init_patterns': AVAILABLE_PATTERNS,
        'current_config': {
            'user_filter': f"{config['user_filter_design']} ({config['user_init_filter']})",
            'item_filter': f"{config['item_filter_design']} ({config['item_init_filter']})",
            'bipartite_filter': f"{config['bipartite_filter_design']} ({config['bipartite_init_filter']})"
        }
    }

def get_quick_configs():
    """Get predefined quick configurations"""
    return {
        'fast': {
            'description': 'Fast training with basic filters',
            'user_filter_design': 'original',
            'item_filter_design': 'original',
            'bipartite_filter_design': 'original',
            'filter_order': 4,
            'epochs': 30
        },
        'balanced': {
            'description': 'Balanced performance with mixed filters',
            'user_filter_design': 'enhanced_basis',
            'item_filter_design': 'chebyshev',
            'bipartite_filter_design': 'original',
            'filter_order': 6,
            'epochs': 50
        },
        'quality': {
            'description': 'High quality with advanced filters',
            'user_filter_design': 'enhanced_basis',
            'item_filter_design': 'parametric',
            'bipartite_filter_design': 'bandstop',
            'filter_order': 8,
            'epochs': 80
        },
        'experimental': {
            'description': 'Experimental ensemble approach',
            'user_filter_design': 'ensemble',
            'item_filter_design': 'ensemble',
            'bipartite_filter_design': 'ensemble',
            'filter_order': 7,
            'epochs': 60
        }
    }

# Print configuration summary at startup
if __name__ != '__main__':
    print_config_summary()