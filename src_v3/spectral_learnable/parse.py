'''
Unified Spectral CF Arguments - Clean and Consistent
Synchronized with static model parameters

@author: Tseesuren Batsuuri
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Spectral CF - Learnable Version")

    # Basic parameters
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-3, help="weight decay")
    parser.add_argument('--train_batch', type=int, default=1000, help='training user batch size')
    parser.add_argument('--test_batch', type=int, default=500, help="evaluation user batch size")
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='ml-100k', help="dataset: ml-100k, lastfm, gowalla, yelp2018, amazon-book")
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    
    # Model configuration  
    parser.add_argument('--model', type=str, default='spectral-cf', help='model name')
    parser.add_argument('--in_mat', type=str, default='uib', help='input matrix: u, i, ui, b, ub, or uib')
    
    # Eigenvalues (same as static model)
    parser.add_argument('--u', type=int, default=8, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=40, help='item eigenvalues')
    parser.add_argument('--b', type=int, default=60, help='bipartite eigenvalues')
    
    # Filter designs (same as static model)
    filter_choices = ['orig', 'cheby', 'jacobi', 'legendre', 'laguerre', 'hermite', 'bernstein', 'multi', 'band', 'ensemble', 'golden', 'harmonic', 'spectral_basis', 'enhanced_basis']
    parser.add_argument('--uf', type=str, default='spectral_basis', choices=filter_choices, help='user filter')
    parser.add_argument('--if_', type=str, default='spectral_basis', choices=filter_choices, help='item filter') 
    parser.add_argument('--bf', type=str, default='spectral_basis', choices=filter_choices, help='bipartite filter')
    
    # Filter initializations (same as static model)
    init_choices = ['smooth', 'sharp', 'bandpass', 'golden', 'butter', 'gauss', 'stop', 'notch']
    parser.add_argument('--up', type=str, default='smooth', choices=init_choices, help='user pattern')
    parser.add_argument('--ip', type=str, default='step_0.7', choices=init_choices, help='item pattern')
    parser.add_argument('--bp', type=str, default='step_0.7', choices=init_choices, help='bipartite pattern')
    
    # Training control
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='min improvement for early stopping')
    parser.add_argument('--n_epoch_eval', type=int, default=5, help='evaluate every N epochs')
    
    # System
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level')
    parser.add_argument('--full_training', action='store_true', help='use full training data (no validation split)')

    return parser.parse_args()


def validate_args(args):
    """Validate and adjust arguments"""
    
    # Print configuration summary
    if args.verbose > 0:
        print(f"\n📋 Configuration Summary:")
        print(f"   └─ Input matrix: {args.in_mat}")
        print(f"   └─ User filter: {args.uf} ({args.up})")
        print(f"   └─ Item filter: {args.if_} ({args.ip})")
        print(f"   └─ Bipartite filter: {args.bf} ({args.bp})")
        print(f"   └─ Eigenvalues: u={args.u}, i={args.i}, b={args.b}")
    
    return args