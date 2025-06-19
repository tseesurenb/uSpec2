"""
Argument parser for Learnable Spectral CF
Clean and organized with per-view hyperparameters
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Learnable Spectral CF")
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book'],
                       help='dataset to use')
    parser.add_argument('--full_training', action='store_true',
                       help='use full training data without validation split')
    
    # Model architecture
    parser.add_argument('--filter', type=str, default='uib', 
                       help='which views to use: u, i, b, ui, ub, ib, uib')
    parser.add_argument('--filter_type', type=str, default='bernstein',
                       choices=['bernstein', 'chebyshev', 'spectral_basis'],
                       help='type of learnable filter')
    parser.add_argument('--filter_order', type=int, default=8,
                       help='order/complexity of filter')
    
    # Eigenvalues
    parser.add_argument('--u', type=int, default=25, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=200, help='item eigenvalues')
    parser.add_argument('--b', type=int, default=220, help='bipartite eigenvalues')
    
    # Filter initialization
    init_choices = ['original', 'smooth', 'sharp', 'lowpass', 'uniform', 'lowfreq',
                   'linear_dec', 'step_0.5', 'step_0.7', 'step_0.9', 
                   'exp_decay', 'constant_1', 'constant_0.1']
    parser.add_argument('--user_init', type=str, default='smooth', choices=init_choices,
                       help='user filter initialization')
    parser.add_argument('--item_init', type=str, default='sharp', choices=init_choices,
                       help='item filter initialization')
    parser.add_argument('--bipartite_init', type=str, default='smooth', choices=init_choices,
                       help='bipartite filter initialization')
    
    # View-specific learning rates
    parser.add_argument('--user_lr', type=float, default=0.1,
                       help='learning rate for user filter')
    parser.add_argument('--item_lr', type=float, default=0.01,
                       help='learning rate for item filter')
    parser.add_argument('--bipartite_lr', type=float, default=0.05,
                       help='learning rate for bipartite filter')
    
    # View-specific weight decay
    parser.add_argument('--user_decay', type=float, default=1e-4,
                       help='weight decay for user filter')
    parser.add_argument('--item_decay', type=float, default=1e-3,
                       help='weight decay for item filter')
    parser.add_argument('--bipartite_decay', type=float, default=5e-4,
                       help='weight decay for bipartite filter')
    
    # Global training settings
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--train_batch', type=int, default=1000, help='training batch size')
    parser.add_argument('--test_batch', type=int, default=500, help='test batch size')
    parser.add_argument('--neg_ratio', type=int, default=4, help='negative sampling ratio')
    parser.add_argument('--loss', type=str, default='bpr', choices=['bpr', 'mse'],
                       help='loss function: bpr (pairwise) or mse (pointwise)')
    
    # Optimization
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='optimizer type')
    parser.add_argument('--scheduler', type=str, default='none',
                       choices=['none', 'cosine', 'step', 'plateau'],
                       help='learning rate scheduler')
    parser.add_argument('--patience', type=int, default=5,
                       help='early stopping patience')
    
    # Evaluation
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    parser.add_argument('--eval_freq', type=int, default=5,
                       help='evaluate every N epochs')
    
    # System
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='device to use')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level')
    
    # Experiment tracking
    parser.add_argument('--exp_name', type=str, default='default',
                       help='experiment name for logging')
    parser.add_argument('--save_model', action='store_true',
                       help='save best model')
    parser.add_argument('--log_filters', action='store_true',
                       help='log filter responses during training')
    
    # Two-hop propagation
    parser.add_argument('--use_two_hop', action='store_true',
                       help='enable two-hop propagation (like GF-CF)')
    parser.add_argument('--two_hop_weight', type=float, default=1.3,
                       help='weight for two-hop propagation (ignored for amazon-book)')
    parser.add_argument('--raw_only', action='store_true',
                       help='use only raw two-hop propagation, no spectral filtering')
    
    # Learnable normalization (PolyCF inspired)
    parser.add_argument('--learnable_gamma', action='store_true',
                       help='enable learnable gamma parameter for item view normalization')
    
    return parser.parse_args()


def get_config(args):
    """Convert args to config dict"""
    import torch
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    config = {
        # Dataset
        'dataset': args.dataset,
        'full_training': args.full_training,
        
        # Model
        'filter': args.filter,
        'filter_type': args.filter_type,
        'filter_order': args.filter_order,
        
        # Eigenvalues
        'u_n_eigen': args.u,
        'i_n_eigen': args.i,
        'b_n_eigen': args.b,
        
        # Initialization
        'user_init': args.user_init,
        'item_init': args.item_init,
        'bipartite_init': args.bipartite_init,
        
        # View-specific hyperparameters
        'user_lr': args.user_lr,
        'item_lr': args.item_lr,
        'bipartite_lr': args.bipartite_lr,
        'user_decay': args.user_decay,
        'item_decay': args.item_decay,
        'bipartite_decay': args.bipartite_decay,
        
        # Training
        'epochs': args.epochs,
        'train_batch_size': args.train_batch,
        'test_batch_size': args.test_batch,
        'neg_ratio': args.neg_ratio,
        'loss': args.loss,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'patience': args.patience,
        
        # Evaluation
        'topks': eval(args.topks),
        'eval_freq': args.eval_freq,
        
        # System
        'seed': args.seed,
        'device': device,
        'verbose': args.verbose,
        
        # Experiment
        'exp_name': args.exp_name,
        'save_model': args.save_model,
        'log_filters': args.log_filters,
        
        # Two-hop propagation
        'use_two_hop': args.use_two_hop,
        'two_hop_weight': args.two_hop_weight,
        'raw_only': args.raw_only,
        
        # Learnable normalization
        'learnable_gamma': args.learnable_gamma,
    }
    
    return config