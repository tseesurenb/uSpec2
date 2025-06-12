'''
Created on June 12, 2025
Updated argument parser with BPR loss and advanced training options
Based on DySimGCF's successful parameters

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Improved Universal Spectral CF with BPR Loss")

    # Basic parameters
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--train_u_batch_size', type=int, default=1024, help='training batch size')
    parser.add_argument('--eval_u_batch_size', type=int, default=500, help="evaluation batch size")
    parser.add_argument('--epochs', type=int, default=200)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='ml-100k', 
                       help="dataset: ml-100k, lastfm, gowalla, yelp2018, amazon-book")
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    
    # Model configuration
    parser.add_argument('--model', type=str, default='uspec', help='model name')
    parser.add_argument('--filter', type=str, default='ui', help='u, i, ui, b, ub, or uib')
    parser.add_argument('--filter_order', type=int, default=6, help='polynomial order')
    
    # Embedding configuration
    parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
    
    # Eigenvalues
    parser.add_argument('--u_n_eigen', type=int, default=0, help='user eigenvalues (0=auto)')
    parser.add_argument('--i_n_eigen', type=int, default=0, help='item eigenvalues (0=auto)')
    parser.add_argument('--b_n_eigen', type=int, default=0, help='bipartite eigenvalues (0=auto)')
    
    # BPR Loss parameters (borrowed from DySimGCF)
    parser.add_argument('--samples', type=int, default=50, help='number of negative samples')
    parser.add_argument('--margin', type=float, default=0.03, help='BPR margin')
    parser.add_argument('--r_loss_w', type=float, default=1.0, help='regularization loss weight')
    
    # Simplified filter designs
    filter_choices = ['simple', 'original', 'enhanced_basis', 'chebyshev']
    
    parser.add_argument('--user_filter_design', type=str, default='simple', choices=filter_choices, help='user filter design')
    parser.add_argument('--item_filter_design', type=str, default='simple', choices=filter_choices, help='item filter design')
    parser.add_argument('--bipartite_filter_design', type=str, default='simple', choices=filter_choices, help='bipartite filter design')
    
    # Initialization patterns
    init_choices = ['smooth', 'sharp', 'bandpass', 'golden_036']
    
    parser.add_argument('--user_init_filter', type=str, default='smooth', choices=init_choices, help='user init pattern')
    parser.add_argument('--item_init_filter', type=str, default='sharp', choices=init_choices, help='item init pattern')
    parser.add_argument('--bipartite_init_filter', type=str, default='smooth', choices=init_choices, help='bipartite init pattern')
    
    # Training control
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='min improvement')
    parser.add_argument('--n_epoch_eval', type=int, default=5, help='eval every N epochs')
    
    # System
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level')
    
    # Quick presets based on DySimGCF findings
    parser.add_argument('--preset', type=str, default=None, 
                       choices=['ml100k_optimized', 'yelp_optimized', 'amazon_optimized', 'fast'], 
                       help='Use dataset-specific optimized configurations')

    return parser.parse_args()


def apply_preset(args):
    """Apply optimized presets based on DySimGCF findings"""
    if args.preset == 'ml100k_optimized':
        # Optimized for ML-100K (based on DySimGCF best params)
        args.lr = 0.001
        args.decay = 1e-3
        args.samples = 50
        args.margin = 0.03
        args.embed_dim = 64
        args.filter = 'ui'
        args.epochs = 200
        args.train_u_batch_size = 1024
        args.filter_order = 6
        print("🎯 Applied ML-100K OPTIMIZED preset: lr=0.001, decay=1e-3, samples=50, margin=0.03")
        
    elif args.preset == 'yelp_optimized':
        # Optimized for Yelp2018
        args.lr = 0.001
        args.decay = 1e-4
        args.samples = 100
        args.margin = 0.1
        args.embed_dim = 64
        args.filter = 'ui'
        args.epochs = 150
        args.train_u_batch_size = 2048
        print("🎯 Applied YELP2018 OPTIMIZED preset")
        
    elif args.preset == 'amazon_optimized':
        # Optimized for Amazon-Book
        args.lr = 0.001
        args.decay = 1e-5
        args.samples = 40
        args.margin = 0.1
        args.embed_dim = 128
        args.filter = 'ui'
        args.epochs = 120
        args.train_u_batch_size = 2048
        print("🎯 Applied AMAZON-BOOK OPTIMIZED preset")
        
    elif args.preset == 'fast':
        # Fast training preset
        args.embed_dim = 32
        args.filter_order = 4
        args.epochs = 50
        args.samples = 10
        args.train_u_batch_size = 2048
        print("🚀 Applied FAST preset: reduced dimensions and epochs")
    
    return args


def validate_args(args):
    """Validate and adjust arguments"""
    
    # Apply preset FIRST before other validations
    if args.preset:
        args = apply_preset(args)
    
    # Validate eigenvalue settings AFTER preset
    if args.u_n_eigen == 0:
        args.u_n_eigen = None  # Auto-calculate
    if args.i_n_eigen == 0:
        args.i_n_eigen = None  # Auto-calculate
    if args.b_n_eigen == 0:
        args.b_n_eigen = None  # Auto-calculate
    
    # Ensure reasonable values
    args.filter_order = max(2, min(args.filter_order, 12))
    args.embed_dim = max(16, min(args.embed_dim, 256))
    args.samples = max(1, min(args.samples, 200))
    args.margin = max(0.01, min(args.margin, 1.0))
    
    # Print configuration summary
    if args.verbose > 0:
        print(f"\n📋 Enhanced Spectral CF Configuration:")
        print(f"   └─ Filter combination: {args.filter}")
        print(f"   └─ Embedding dimension: {args.embed_dim}")
        print(f"   └─ Filter order: {args.filter_order}")
        print(f"   └─ Negative samples: {args.samples}")
        print(f"   └─ BPR margin: {args.margin}")
        print(f"   └─ Learning rate: {args.lr}")
        print(f"   └─ Weight decay: {args.decay}")
    
    return args