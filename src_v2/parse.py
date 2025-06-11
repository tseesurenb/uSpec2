'''
Created on June 7, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Enhanced argument parser with user-specific model support

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Universal Spectral model for CF with user-specific model selection")

    # Basic training parameters
    parser.add_argument('--lr', type=float, default=0.1, help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-3, help="the weight decay for l2 normalizaton")
    parser.add_argument('--train_u_batch_size', type=int, default=1000, help='batch size for training users, -1 for full dataset')
    parser.add_argument('--eval_u_batch_size', type=int, default=500, help="batch size for evaluation users (memory management)")
    parser.add_argument('--epochs', type=int, default=50)
    
    # Dataset and evaluation
    parser.add_argument('--dataset', type=str, default='gowalla', help="available datasets: [lastfm, gowalla, yelp2018, amazon-book, ml-100k]")
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of training data to use for validation (0.1 = 10%)')
    
    # Model selection and architecture - UPDATED WITH USER-SPECIFIC SUPPORT
    parser.add_argument('--model_type', type=str, default='enhanced', 
                       choices=['basic', 'enhanced', 'simple', 'user_specific'],
                       help='Model type: basic (model.py), enhanced (model_enhanced.py), simple (simple_model.py), or user_specific (model_user_specific.py)')
    parser.add_argument('--n_eigen', type=int, default=0, 
                       help='Number of eigenvalues for both user and item matrices (legacy, use u_n_eigen/i_n_eigen instead)')
    parser.add_argument('--u_n_eigen', type=int, default=0, 
                       help='Number of eigenvalues for user similarity matrix (0 = auto-adaptive)')
    parser.add_argument('--i_n_eigen', type=int, default=0, 
                       help='Number of eigenvalues for item similarity matrix (0 = auto-adaptive)')
    parser.add_argument('--b_n_eigen', type=int, default=0, 
                       help='Number of eigenvalues for bipartite user-item matrix (0 = auto-adaptive, only for enhanced model with filter=uib)')
    parser.add_argument('--model', type=str, default='uspec', help='rec-model, support [uspec]')
    parser.add_argument('--m_type', type=str, default='single', help='single or double similarity')
    parser.add_argument('--filter', type=str, default='uib', help='u, i, ui, b, or uib (uib = three-view filtering)')
    parser.add_argument('--filter_order', type=int, default=6, help='polynomial order for spectral filters')
    
    # Filter design options (only for enhanced model)
    parser.add_argument('--init_filter', type=str, default='smooth', help='Initial filter pattern (only used with enhanced model)')
    
    # Simple model specific parameters
    parser.add_argument('--filter_mode', type=str, default='single', 
                       choices=['single', 'dual'],
                       help='Filter mode for simple model: single or dual filters')
    parser.add_argument('--n_hops', type=int, default=2,
                       choices=[1, 2],
                       help='Number of hops for simple model: 1 (User→Item) or 2 (User→Item→User→Item)')
    
    # User-specific model parameters (NEW!)
    parser.add_argument('--shared_base', action='store_true', default=True,
                       help='Use shared base filter with user-specific adaptations (user_specific model only)')
    parser.add_argument('--no_shared_base', dest='shared_base', action='store_false',
                       help='Use fully personalized filter parameters per user (user_specific model only)')
    parser.add_argument('--personalization_dim', type=int, default=8,
                       help='Dimensionality of user personalization embeddings (user_specific model only)')
    parser.add_argument('--cold_start_strategy', type=str, default='average',
                       choices=['average', 'random', 'base'],
                       help='Strategy for handling new users (user_specific model only)')
    
    # Laplacian-specific parameters
    parser.add_argument('--use_laplacian', action='store_true', default=False, help='Use Laplacian-based spectral filtering')
    parser.add_argument('--no_laplacian', dest='use_laplacian', action='store_false', help='Use similarity-based approach')
    parser.add_argument('--laplacian_type', type=str, default='normalized_sym',
                       choices=['unnormalized', 'normalized_sym', 'normalized_rw'],
                       help='Type of Laplacian matrix')
    
    # Enhanced similarity-aware parameters
    parser.add_argument('--use_similarity_norm', action='store_true', default=False, 
                       help='Use similarity-weighted normalization instead of degree-based')
    parser.add_argument('--similarity_type', type=str, default='cosine', 
                       choices=['cosine', 'jaccard'], help='Similarity measure for enhanced model')
    parser.add_argument('--similarity_threshold', type=float, default=0.01,
                       help='Threshold for filtering weak similarities (maintains symmetry)')
    parser.add_argument('--similarity_weight', type=float, default=0.8,
                       help='Weight for similarity vs original adjacency (0.0-1.0)')
    
    # Training control
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='minimum improvement for early stopping')
    parser.add_argument('--n_epoch_eval', type=int, default=5, help='evaluate every N epochs')
    
    # Experiment control
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='0 for silent, 1 for verbose')

    parser.add_argument('--filter_design', type=str, default='multiscale', 
                   choices=['original', 'basis', 'enhanced_basis', 'adaptive_golden', 'adaptive', 
                           'neural', 'deep', 'multiscale', 'ensemble',
                           'band_stop', 'adaptive_band_stop', 'parametric_multi_band', 'harmonic',
                           # NEW POLYNOMIAL FILTERS:
                           'polynomial', 'chebyshev', 'jacobi', 'legendre', 'adaptive_polynomial'], 
                   help='Filter design (only used with enhanced model)')

    # NEW POLYNOMIAL-SPECIFIC ARGUMENTS:
    parser.add_argument('--polynomial_type', type=str, default='chebyshev',
                    choices=['chebyshev', 'jacobi', 'legendre', 'laguerre', 'hermite', 'bernstein'],
                    help='Polynomial type for generic polynomial filter')

    parser.add_argument('--polynomial_alpha', type=float, default=0.0,
                    help='Alpha parameter for Jacobi/Laguerre polynomials')

    parser.add_argument('--polynomial_beta', type=float, default=0.0,
                    help='Beta parameter for Jacobi polynomials')

    parser.add_argument('--adaptive_polynomial_types', type=str, nargs='+', 
                    default=['chebyshev', 'legendre', 'jacobi'],
                    help='Polynomial types for adaptive polynomial filter')

    
    return parser.parse_args()


def validate_args(args):
    """Enhanced argument validation with user-specific model support"""
    
    # Model type validation and recommendations
    if args.model_type == 'basic':
        print(f"📦 Using basic model (model.py)")
        if args.filter_design != 'enhanced_basis':
            print(f"   Note: filter_design and init_filter are ignored with basic model")
        if args.filter in ['b', 'uib']:
            print(f"   ⚠️  Warning: Bipartite filtering (b/uib) not supported in basic model")
            print(f"   Suggestion: Use enhanced model for three-view filtering")
            
    elif args.model_type == 'simple':
        print(f"⚡ Using simple model (simple_model.py)")
        print(f"   Filter Mode: {args.filter_mode}")
        print(f"   Fast and minimal implementation")
        if args.filter in ['b', 'uib']:
            print(f"   ⚠️  Warning: Bipartite filtering (b/uib) not supported in simple model")
            print(f"   Suggestion: Use enhanced model for three-view filtering")
            
    elif args.model_type == 'user_specific':
        print(f"🎯 Using user-specific model (model_user_specific.py)")
        print(f"   Shared Base: {args.shared_base}")
        print(f"   Personalization Dimension: {args.personalization_dim}")
        print(f"   Cold Start Strategy: {args.cold_start_strategy}")
        
        # User-specific model validations
        if args.personalization_dim < 2:
            print(f"   ⚠️  Warning: Very low personalization_dim ({args.personalization_dim})")
            print(f"   Suggestion: Use at least 4-8 dimensions for meaningful personalization")
        elif args.personalization_dim > 64:
            print(f"   ⚠️  Warning: High personalization_dim ({args.personalization_dim}) may cause overfitting")
            print(f"   Suggestion: Consider 8-32 dimensions for most datasets")
        
        if args.filter in ['b', 'uib']:
            print(f"   ⚠️  Warning: Bipartite filtering (b/uib) not fully tested with user-specific model")
            print(f"   Current implementation focuses on user and item similarity filtering")
            
    else:
        print(f"🚀 Using enhanced model (model_enhanced.py)")
        print(f"   Filter Design: {args.filter_design}")
        print(f"   Init Filter: {args.init_filter}")
        
        # NEW: Polynomial filter validation (ADDITION)
        polynomial_filters = ['polynomial', 'chebyshev', 'jacobi', 'legendre', 'adaptive_polynomial']
        if args.filter_design in polynomial_filters:
            print(f"   🔢 POLYNOMIAL FILTER ENABLED")
            if args.filter_design == 'polynomial':
                print(f"      Polynomial Type: {args.polynomial_type}")
            elif args.filter_design == 'jacobi':
                print(f"      Jacobi Parameters: α={args.polynomial_alpha}, β={args.polynomial_beta}")
            elif args.filter_design == 'adaptive_polynomial':
                print(f"      Adaptive Types: {args.adaptive_polynomial_types}")
        
        # Three-view validation
        if args.filter in ['b', 'uib']:
            print(f"   🔍 THREE-VIEW FILTERING ENABLED")
            if args.filter == 'uib':
                print(f"      All three views: User-User + Item-Item + Bipartite")
            else:
                print(f"      Bipartite view only")
    
    # Eigenvalue parameter validation with bipartite support
    if args.u_n_eigen > 0 and args.i_n_eigen > 0:
        print(f"✅ Using separate eigenvalue counts: u_n_eigen={args.u_n_eigen}, i_n_eigen={args.i_n_eigen}")
        
        if args.b_n_eigen > 0:
            print(f"   🔗 Bipartite eigenvalues: b_n_eigen={args.b_n_eigen}")
            if args.filter not in ['b', 'uib']:
                print(f"   Note: b_n_eigen specified but filter={args.filter} doesn't use bipartite view")
        elif args.filter in ['b', 'uib']:
            print(f"   🤖 Bipartite eigenvalues: Auto-adaptive (recommended for three-view)")
            
        if args.n_eigen > 0:
            print(f"   Note: n_eigen={args.n_eigen} will be ignored in favor of separate eigenvalue counts")
            
    elif args.n_eigen > 0:
        print(f"📝 Using legacy n_eigen={args.n_eigen} for user and item matrices")
        if args.b_n_eigen > 0:
            print(f"   🔗 Bipartite eigenvalues: b_n_eigen={args.b_n_eigen}")
        elif args.filter in ['b', 'uib']:
            print(f"   🤖 Bipartite eigenvalues: Auto-adaptive")
        if args.model_type in ['enhanced', 'simple', 'user_specific']:
            print(f"   Recommendation: Use --u_n_eigen and --i_n_eigen for better performance")
            
    else:
        if args.model_type == 'simple':
            print(f"🤖 Using default eigenvalue counts (128 each)")
        elif args.model_type in ['enhanced', 'user_specific']:
            print(f"🤖 Using auto-adaptive eigenvalue counts (recommended)")
            if args.filter in ['b', 'uib']:
                print(f"   🔗 Including auto-adaptive bipartite eigenvalues")
        else:
            print(f"📝 Using default eigenvalue counts")
    
    # Dataset-specific recommendations with user-specific support
    dataset_recommendations = {
        'ml-100k': {'u_n_eigen': 48, 'i_n_eigen': 64, 'b_n_eigen': 80, 'personalization_dim': 8},
        'ml-1m': {'u_n_eigen': 96, 'i_n_eigen': 128, 'b_n_eigen': 160, 'personalization_dim': 16},
        'lastfm': {'u_n_eigen': 64, 'i_n_eigen': 96, 'b_n_eigen': 120, 'personalization_dim': 12},
        'gowalla': {'u_n_eigen': 128, 'i_n_eigen': 256, 'b_n_eigen': 200, 'personalization_dim': 24},
        'yelp2018': {'u_n_eigen': 192, 'i_n_eigen': 384, 'b_n_eigen': 300, 'personalization_dim': 32},
        'amazon-book': {'u_n_eigen': 256, 'i_n_eigen': 512, 'b_n_eigen': 400, 'personalization_dim': 32}
    }
    
    if args.dataset in dataset_recommendations:
        rec = dataset_recommendations[args.dataset]
        if args.u_n_eigen == 0 and args.i_n_eigen == 0 and args.n_eigen == 0:
            if args.model_type == 'user_specific':
                print(f"💡 {args.dataset} user-specific recommendations:")
                print(f"   --u_n_eigen {rec['u_n_eigen']} --i_n_eigen {rec['i_n_eigen']} --personalization_dim {rec['personalization_dim']}")
            elif args.filter == 'uib':
                print(f"💡 {args.dataset} three-view recommendation:")
                print(f"   --u_n_eigen {rec['u_n_eigen']} --i_n_eigen {rec['i_n_eigen']} --b_n_eigen {rec['b_n_eigen']}")
            else:
                print(f"💡 {args.dataset} recommendation: --u_n_eigen {rec['u_n_eigen']} --i_n_eigen {rec['i_n_eigen']}")
    
    # Model compatibility checks
    if args.filter in ['b', 'uib'] and args.model_type not in ['enhanced']:
        print(f"❌ ERROR: Bipartite filtering (--filter {args.filter}) requires enhanced model")
        print(f"   Solution: Add --model_type enhanced")
        return None
    
    # User-specific model parameter validation
    if args.model_type == 'user_specific':
        if args.personalization_dim <= 0:
            print(f"❌ ERROR: personalization_dim must be positive for user_specific model")
            return None
        
        if args.cold_start_strategy not in ['average', 'random', 'base']:
            print(f"❌ ERROR: Invalid cold_start_strategy. Choose from: average, random, base")
            return None
    
    return args


if __name__ == "__main__":
    # Demo the new user-specific functionality
    print("=== USER-SPECIFIC MODEL EXAMPLES ===")
    print()
    print("# User-specific model with shared base:")
    print("python main.py --model_type user_specific --dataset ml-100k --shared_base --personalization_dim 8")
    print()
    print("# User-specific model fully personalized:")
    print("python main.py --model_type user_specific --dataset ml-100k --no_shared_base")
    print()
    print("# User-specific with custom parameters:")
    print("python main.py --model_type user_specific --dataset gowalla --u_n_eigen 128 --i_n_eigen 256 --personalization_dim 24 --cold_start_strategy average")
    print()
    print("# Three-view model (User + Item + Bipartite):")
    print("python main.py --model_type enhanced --dataset ml-100k --filter uib --u_n_eigen 48 --i_n_eigen 64 --b_n_eigen 80")
    print()
    print("# Auto-adaptive user-specific (recommended):")
    print("python main.py --model_type user_specific --dataset ml-100k")