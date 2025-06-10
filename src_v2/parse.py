'''
Created on June 7, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Enhanced argument parser with three-view bipartite support

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Universal Spectral model for CF with model selection")

    # Basic training parameters
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-2, help="the weight decay for l2 normalizaton")
    parser.add_argument('--train_u_batch_size', type=int, default=1000, help='batch size for training users, -1 for full dataset')
    parser.add_argument('--eval_u_batch_size', type=int, default=500, help="batch size for evaluation users (memory management)")
    parser.add_argument('--epochs', type=int, default=50)
    
    # Dataset and evaluation
    parser.add_argument('--dataset', type=str, default='gowalla', help="available datasets: [lastfm, gowalla, yelp2018, amazon-book, ml-100k]")
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of training data to use for validation (0.1 = 10%)')
    
    # Model selection and architecture - UPDATED WITH BIPARTITE SUPPORT
    parser.add_argument('--model_type', type=str, default='enhanced', 
                       choices=['basic', 'enhanced', 'simple'],
                       help='Model type: basic (model.py), enhanced (model_enhanced.py), or simple (simple_model.py)')
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
    parser.add_argument('--filter', type=str, default='ui', help='u, i, ui, b, or uib (uib = three-view filtering)')
    parser.add_argument('--filter_order', type=int, default=6, help='polynomial order for spectral filters')
    
    # Filter design options (only for enhanced model)
    parser.add_argument('--filter_design', type=str, default='enhanced_basis', 
                       choices=['original', 'basis', 'enhanced_basis', 'adaptive_golden', 'adaptive', 'neural', 'deep', 'multiscale', 'ensemble',
                               'band_stop', 'adaptive_band_stop', 'parametric_multi_band', 'harmonic'], 
                       help='Filter design (only used with enhanced model)')
    parser.add_argument('--init_filter', type=str, default='smooth', help='Initial filter pattern (only used with enhanced model)')
    
    # Simple model specific parameters
    parser.add_argument('--filter_mode', type=str, default='single', 
                       choices=['single', 'dual'],
                       help='Filter mode for simple model: single or dual filters')
    parser.add_argument('--n_hops', type=int, default=2,
                       choices=[1, 2],
                       help='Number of hops for simple model: 1 (User→Item) or 2 (User→Item→User→Item)')
    
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
    
    return parser.parse_args()


def validate_args(args):
    """Enhanced argument validation with three-view support"""
    
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
            
    else:
        print(f"🚀 Using enhanced model (model_enhanced.py)")
        print(f"   Filter Design: {args.filter_design}")
        print(f"   Init Filter: {args.init_filter}")
        
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
        if args.model_type in ['enhanced', 'simple']:
            print(f"   Recommendation: Use --u_n_eigen and --i_n_eigen for better performance")
            
    else:
        if args.model_type == 'simple':
            print(f"🤖 Using default eigenvalue counts (128 each)")
        elif args.model_type == 'enhanced':
            print(f"🤖 Using auto-adaptive eigenvalue counts (recommended)")
            if args.filter in ['b', 'uib']:
                print(f"   🔗 Including auto-adaptive bipartite eigenvalues")
        else:
            print(f"📝 Using default eigenvalue counts")
    
    # Dataset-specific recommendations with bipartite support
    dataset_recommendations = {
        'ml-100k': {'u_n_eigen': 48, 'i_n_eigen': 64, 'b_n_eigen': 80},
        'ml-1m': {'u_n_eigen': 96, 'i_n_eigen': 128, 'b_n_eigen': 160},
        'lastfm': {'u_n_eigen': 64, 'i_n_eigen': 96, 'b_n_eigen': 120},
        'gowalla': {'u_n_eigen': 128, 'i_n_eigen': 256, 'b_n_eigen': 200},
        'yelp2018': {'u_n_eigen': 192, 'i_n_eigen': 384, 'b_n_eigen': 300},
        'amazon-book': {'u_n_eigen': 256, 'i_n_eigen': 512, 'b_n_eigen': 400}
    }
    
    if args.dataset in dataset_recommendations:
        rec = dataset_recommendations[args.dataset]
        if args.u_n_eigen == 0 and args.i_n_eigen == 0 and args.n_eigen == 0:
            if args.filter == 'uib':
                print(f"💡 {args.dataset} three-view recommendation:")
                print(f"   --u_n_eigen {rec['u_n_eigen']} --i_n_eigen {rec['i_n_eigen']} --b_n_eigen {rec['b_n_eigen']}")
            else:
                print(f"💡 {args.dataset} recommendation: --u_n_eigen {rec['u_n_eigen']} --i_n_eigen {rec['i_n_eigen']}")
    
    # Filter compatibility check
    if args.filter in ['b', 'uib'] and args.model_type != 'enhanced':
        print(f"❌ ERROR: Bipartite filtering (--filter {args.filter}) requires enhanced model")
        print(f"   Solution: Add --model_type enhanced")
        return None
    
    return args


if __name__ == "__main__":
    # Demo the new three-view functionality
    print("=== THREE-VIEW MODEL EXAMPLES ===")
    print()
    print("# Three-view model (User + Item + Bipartite):")
    print("python main.py --model_type enhanced --dataset ml-100k --filter uib --u_n_eigen 48 --i_n_eigen 64 --b_n_eigen 80")
    print()
    print("# Bipartite-only model:")
    print("python main.py --model_type enhanced --dataset ml-100k --filter b --b_n_eigen 100")
    print()
    print("# Auto-adaptive three-view (recommended):")
    print("python main.py --model_type enhanced --dataset ml-100k --filter uib")
    print()
    print("# Traditional two-view model:")
    print("python main.py --model_type enhanced --dataset ml-100k --filter ui --u_n_eigen 48 --i_n_eigen 64")