'''
Unified Spectral CF Arguments - Clean and Consistent
Synchronized with learnable model parameters

@author: Tseesuren Batsuuri
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Spectral CF - Static Version")
    
    # Basic parameters (keep legacy for compatibility)
    parser.add_argument('--test_batch', type=int, default=500, help="evaluation user batch size")
    parser.add_argument('--testbatch', type=int, default=500, help="legacy: evaluation user batch size")
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='ml-100k',
                        help="dataset: ml-100k, lastfm, gowalla, yelp2018, amazon-book")
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    
    # Model configuration
    parser.add_argument('--model', type=str, default='spectral-cf', help='model name')
    parser.add_argument('--in_mat', type=str, default='ui', help='input matrix: u, i, ui, b, ub, or uib')
    
    # Eigenvalues (same as learnable model)
    parser.add_argument('--u', type=int, default=150, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=400, help='item eigenvalues')
    parser.add_argument('--b', type=int, default=800, help='bipartite eigenvalues')
    
    # Filter designs (same as learnable model)
    filter_choices = ['orig', 'cheby', 'jacobi', 'legendre', 'laguerre', 'hermite', 'bernstein', 
                     'multi', 'band', 'ensemble', 'golden', 'harmonic', 'spectral_basis', 'enhanced_basis']
    parser.add_argument('--uf', type=str, default='orig', choices=filter_choices, help='user filter')
    parser.add_argument('--if', type=str, default='orig', choices=filter_choices, help='item filter') 
    parser.add_argument('--bf', type=str, default='orig', choices=filter_choices, help='bipartite filter')
    
    # Filter initializations (same as learnable model)
    init_choices = ['smooth', 'sharp', 'bandpass', 'golden', 'butter', 'gauss', 'stop', 'notch']
    parser.add_argument('--up', type=str, default='smooth', choices=init_choices, help='user pattern')
    parser.add_argument('--ip', type=str, default='sharp', choices=init_choices, help='item pattern')
    parser.add_argument('--bp', type=str, default='smooth', choices=init_choices, help='bipartite pattern')
    
    # System
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level')
    
    # Legacy parameters (keep for compatibility)
    parser.add_argument('--path', type=str, default="./checkpoints", help="legacy: path to save weights")
    parser.add_argument('--comment', type=str, default="spectral-cf", help="legacy: comment")
    parser.add_argument('--load', type=int, default=0, help="legacy: load weights")
    parser.add_argument('--epochs', type=int, default=0, help="legacy: training epochs")
    parser.add_argument('--multicore', type=int, default=0, help='legacy: multiprocessing in test')
    parser.add_argument('--simple_model', type=str, default='none', help='legacy: simple model')
    
    return parser.parse_args()


def validate_args(args):
    """Validate and adjust arguments"""
    
    # Handle legacy parameter mapping
    if hasattr(args, 'testbatch') and args.testbatch != 500:
        args.test_batch = args.testbatch
    
    # Print configuration summary
    if args.verbose > 0:
        print(f"\n📋 Configuration Summary:")
        print(f"   └─ Input matrix: {args.in_mat}")
        print(f"   └─ User filter: {args.uf} ({args.up})")
        print(f"   └─ Item filter: {getattr(args, 'if')} ({args.ip})")
        print(f"   └─ Bipartite filter: {args.bf} ({args.bp})")
        print(f"   └─ Eigenvalues: u={args.u}, i={args.i}, b={args.b}")
    
    return args