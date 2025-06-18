"""
Configuration for v2 - using complete v1 config to avoid missing keys
"""
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Raw Symmetric Softmax CF")
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='gowalla',
                       choices=['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book'])
    parser.add_argument('--full_training', action='store_true',
                       help='use full training data without validation split')
    
    # Model architecture (keeping for compatibility)
    parser.add_argument('--filter', type=str, default='i', 
                       help='fixed to i for v2')
    parser.add_argument('--filter_type', type=str, default='bernstein',
                       choices=['bernstein', 'chebyshev', 'spectral_basis'])
    parser.add_argument('--filter_order', type=int, default=8)
    
    # Eigenvalues (keeping for compatibility)
    parser.add_argument('--u', type=int, default=25)
    parser.add_argument('--i', type=int, default=200)
    parser.add_argument('--b', type=int, default=220)
    
    # Filter initialization (keeping for compatibility)
    init_choices = ['original', 'smooth', 'sharp', 'lowpass', 'uniform', 'lowfreq',
                   'linear_dec', 'step_0.5', 'step_0.7', 'step_0.9', 
                   'exp_decay', 'constant_1', 'constant_0.1']
    parser.add_argument('--user_init', type=str, default='smooth', choices=init_choices)
    parser.add_argument('--item_init', type=str, default='sharp', choices=init_choices)
    parser.add_argument('--bipartite_init', type=str, default='smooth', choices=init_choices)
    
    # View-specific learning rates (keeping for compatibility)
    parser.add_argument('--user_lr', type=float, default=0.1)
    parser.add_argument('--item_lr', type=float, default=0.01)
    parser.add_argument('--bipartite_lr', type=float, default=0.05)
    
    # View-specific weight decay (keeping for compatibility)
    parser.add_argument('--user_decay', type=float, default=1e-4)
    parser.add_argument('--item_decay', type=float, default=1e-3)
    parser.add_argument('--bipartite_decay', type=float, default=5e-4)
    
    # Training
    parser.add_argument('--epochs', type=int, default=1, help='not used in v2')
    parser.add_argument('--train_batch', type=int, default=1000)
    parser.add_argument('--test_batch', type=int, default=500)
    parser.add_argument('--neg_ratio', type=int, default=4)
    parser.add_argument('--loss', type=str, default='mse', choices=['bpr', 'mse'])
    
    # V2 specific
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for softmax')
    
    # Optimization (keeping for compatibility)
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='none',
                       choices=['none', 'cosine', 'step', 'plateau'])
    parser.add_argument('--patience', type=int, default=5)
    
    # Evaluation
    parser.add_argument('--topks', nargs='?', default="[20]")
    parser.add_argument('--eval_freq', type=int, default=5)
    
    # System
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--verbose', type=int, default=1)
    
    # Experiment tracking (keeping for compatibility)
    parser.add_argument('--exp_name', type=str, default='v2_symmetric_softmax')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--log_filters', action='store_true')
    
    # Two-hop propagation (keeping for compatibility)
    parser.add_argument('--use_two_hop', action='store_true')
    parser.add_argument('--two_hop_weight', type=float, default=0.3)
    parser.add_argument('--raw_only', action='store_true')
    
    return parser.parse_args()


def get_config(args):
    """Convert args to config dict - complete compatibility with v1"""
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
        
        # V2 specific
        'temperature': args.temperature,
    }
    
    return config