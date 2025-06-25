'''
Unified Spectral CF World Configuration
Clean and synchronized with static model

@author: Tseesuren Batsuuri
'''

import os
import torch
from config import parse_args, get_config
import multiprocessing

# Parse arguments
args = parse_args()
config = get_config(args)

# Legacy variables for compatibility
device = config['device']
CORES = multiprocessing.cpu_count() // 2
seed = config['seed']
dataset = config['dataset']
topks = config['topks']

def cprint(words: str):
    """Colored print for dataset loading"""
    print(f"\033[0;30;43m{words}\033[0m")

def print_config_summary():
    """Print a nice configuration summary"""
    if config['verbose'] > 0:
        print(f"\n🎛️  \033[96mSpectral CF Configuration\033[0m")
        print(f"┌─ Dataset: \033[94m{dataset}\033[0m")
        print(f"├─ Device: \033[94m{device}\033[0m")
        print(f"├─ Eigenvalues: \033[94mu={config['u_eigen']}, i={config['i_eigen']}, b={config['b_eigen']}\033[0m")
        print(f"├─ Epochs: \033[94m{config['epochs']}\033[0m")
        print(f"└─ Seed: \033[94m{seed}\033[0m")
        
        print(f"\n🔧 \033[95mLearning Rates:\033[0m")
        print(f"├─ User: \033[93m{config.get('user_lr', 0.01)}\033[0m")
        print(f"├─ Item: \033[93m{config.get('item_lr', 0.01)}\033[0m")
        print(f"└─ Bipartite: \033[93m{config.get('bipartite_lr', 0.01)}\033[0m")

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