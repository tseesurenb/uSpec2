'''
Created on June 12, 2025
Enhanced argument parser with all filter options
Minimalist approach with comprehensive filter support

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Universal Spectral CF - All Filter Types")

    # Basic parameters
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-3, help="weight decay")
    parser.add_argument('--train_u_batch_size', type=int, default=1000, help='training batch size')
    parser.add_argument('--eval_u_batch_size', type=int, default=500, help="evaluation batch size")
    parser.add_argument('--epochs', type=int, default=50)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='ml-100k', 
                       help="dataset: ml-100k, lastfm, gowalla, yelp2018, amazon-book")
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    
    # Model configuration
    parser.add_argument('--model', type=str, default='uspec', help='model name')
    parser.add_argument('--filter', type=str, default='ui', help='u, i, ui, b, ub, or uib')
    parser.add_argument('--filter_order', type=int, default=6, help='polynomial order')
    
    # Eigenvalues
    parser.add_argument('--u_n_eigen', type=int, default=0, help='user eigenvalues (0=auto)')
    parser.add_argument('--i_n_eigen', type=int, default=0, help='item eigenvalues (0=auto)')
    parser.add_argument('--b_n_eigen', type=int, default=0, help='bipartite eigenvalues (0=auto)')
    
    # Filter designs (expanded options)
    filter_choices = [
        'original',              # Basic universal filter
        'spectral_basis',        # Spectral basis filter
        'enhanced_basis',        # Enhanced multi-pattern filter
        'chebyshev',            # Chebyshev polynomial filter
        'jacobi',               # Jacobi polynomial filter
        'legendre',             # Legendre polynomial filter
        'laguerre',             # Laguerre polynomial filter
        'hermite',              # Hermite polynomial filter
        'bernstein',            # Bernstein polynomial filter
        'universal_polynomial',  # Universal polynomial filter
        'bandstop',             # Band-stop filter
        'adaptive_bandstop',    # Advanced multi-band stop filter
        'parametric',           # Parametric multi-band filter
        'multiscale',           # Multi-scale spectral filter
        'harmonic',             # Harmonic series filter
        'golden',               # Adaptive golden ratio filter
        'ensemble'              # Ensemble of all filters
    ]
    
    parser.add_argument('--user_filter_design', type=str, default='multiscale', choices=filter_choices, help='user filter design')
    parser.add_argument('--item_filter_design', type=str, default='chebyshev', choices=filter_choices, help='item filter design')
    parser.add_argument('--bipartite_filter_design', type=str, default='original', choices=filter_choices, help='bipartite filter design')
    
    # Initialization patterns (expanded options)
    init_choices = [
        'smooth', 'sharp', 'bandpass', 'golden_036', 
        'butterworth', 'gaussian', 'band_stop', 'notch'
    ]
    
    parser.add_argument('--user_init_filter', type=str, default='smooth', choices=init_choices, help='user init pattern')
    parser.add_argument('--item_init_filter', type=str, default='sharp', choices=init_choices, help='item init pattern')
    parser.add_argument('--bipartite_init_filter', type=str, default='smooth', choices=init_choices, help='bipartite init pattern')
    
    # Personalization dimensions
    parser.add_argument('--user_personalization_dim', type=int, default=16, help='user personalization dimension')
    parser.add_argument('--item_personalization_dim', type=int, default=12, help='item personalization dimension')
    parser.add_argument('--bipartite_personalization_dim', type=int, default=20, help='bipartite personalization dimension')
    
    # Advanced filter parameters
    parser.add_argument('--n_bands', type=int, default=4, help='number of bands for parametric/multiscale filter')
    parser.add_argument('--n_harmonics', type=int, default=3, help='number of harmonics for harmonic filter')
    parser.add_argument('--n_stop_bands', type=int, default=2, help='number of stop bands for adaptive bandstop filter')
    parser.add_argument('--ensemble_temperature', type=float, default=1.0, help='temperature for ensemble mixing')
    
    # Polynomial filter parameters
    parser.add_argument('--polynomial_type', type=str, default='chebyshev', choices=['chebyshev', 'jacobi', 'legendre', 'laguerre', 'hermite', 'bernstein'],help='polynomial type for universal polynomial filter')
    parser.add_argument('--alpha', type=float, default=0.0, help='alpha parameter for Jacobi/Laguerre polynomials')
    parser.add_argument('--beta', type=float, default=0.0, help='beta parameter for Jacobi polynomials')
    
    # Training control
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='min improvement')
    parser.add_argument('--n_epoch_eval', type=int, default=5, help='eval every N epochs')
    
    # System
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level')
    
    # Quick presets
    parser.add_argument('--preset', type=str, default=None, choices=['fast', 'balanced', 'quality', 'experimental'], help='Use predefined configuration presets')

    return parser.parse_args()


def apply_preset(args):
    """Apply preset configurations"""
    if args.preset == 'fast':
        # Fast training preset
        args.user_filter_design = 'original'
        args.item_filter_design = 'original'
        args.bipartite_filter_design = 'original'
        args.user_personalization_dim = 8
        args.item_personalization_dim = 6
        args.bipartite_personalization_dim = 10
        args.filter_order = 4
        args.epochs = 30
        print("🚀 Applied FAST preset: original filters, small dimensions")
        
    elif args.preset == 'balanced':
        # Balanced performance preset
        args.user_filter_design = 'enhanced_basis'
        args.item_filter_design = 'chebyshev'
        args.bipartite_filter_design = 'original'
        args.user_personalization_dim = 16
        args.item_personalization_dim = 12
        args.bipartite_personalization_dim = 20
        args.filter_order = 6
        print("⚖️ Applied BALANCED preset: mixed filters, standard dimensions")
        
    elif args.preset == 'quality':
        # High quality preset
        args.user_filter_design = 'enhanced_basis'
        args.item_filter_design = 'parametric'
        args.bipartite_filter_design = 'bandstop'
        args.user_personalization_dim = 24
        args.item_personalization_dim = 18
        args.bipartite_personalization_dim = 28
        args.filter_order = 8
        args.epochs = 80
        print("💎 Applied QUALITY preset: advanced filters, large dimensions")
        
    elif args.preset == 'experimental':
        # Experimental preset with ensemble
        args.user_filter_design = 'ensemble'
        args.item_filter_design = 'ensemble'
        args.bipartite_filter_design = 'ensemble'
        args.user_personalization_dim = 20
        args.item_personalization_dim = 16
        args.bipartite_personalization_dim = 24
        args.filter_order = 7
        args.epochs = 60
        print("🧪 Applied EXPERIMENTAL preset: ensemble filters")
    
    return args


def validate_args(args):
    """Validate and adjust arguments"""
    
    # Apply preset if specified
    if args.preset:
        args = apply_preset(args)
    
    # Validate eigenvalue settings
    if args.u_n_eigen == 0:
        args.u_n_eigen = None  # Auto-calculate
    if args.i_n_eigen == 0:
        args.i_n_eigen = None  # Auto-calculate
    if args.b_n_eigen == 0:
        args.b_n_eigen = None  # Auto-calculate
    
    # Ensure reasonable values
    args.filter_order = max(2, min(args.filter_order, 12))
    args.user_personalization_dim = max(4, min(args.user_personalization_dim, 64))
    args.item_personalization_dim = max(4, min(args.item_personalization_dim, 64))
    args.bipartite_personalization_dim = max(4, min(args.bipartite_personalization_dim, 64))
    
    # Print configuration summary
    if args.verbose > 0:
        print(f"\n📋 Configuration Summary:")
        print(f"   └─ Filter combination: {args.filter}")
        print(f"   └─ User filter: {args.user_filter_design} ({args.user_init_filter})")
        print(f"   └─ Item filter: {args.item_filter_design} ({args.item_init_filter})")
        print(f"   └─ Bipartite filter: {args.bipartite_filter_design} ({args.bipartite_init_filter})")
        print(f"   └─ Filter order: {args.filter_order}")
        print(f"   └─ Personalization dims: U={args.user_personalization_dim}, I={args.item_personalization_dim}, B={args.bipartite_personalization_dim}")
    
    return args