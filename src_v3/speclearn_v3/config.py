"""
Minimal configuration for clean spectral CF
"""
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Spectral CF")
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'gowalla', 'yelp2018', 'amazon-book'],
                       help='dataset to use')
    
    # Model - eigenvalues only
    parser.add_argument('--u', type=int, default=64, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=256, help='item eigenvalues') 
    parser.add_argument('--b', type=int, default=256, help='bipartite eigenvalues')
    
    # Model type
    parser.add_argument('--use_laplacian', action='store_true', help='use Laplacian instead of similarity matrices')
    parser.add_argument('--laplacian_type', type=str, default='normalized', 
                       choices=['normalized', 'random_walk'], help='type of Laplacian to use')
    
    # Learning rates per view (with short aliases)
    parser.add_argument('--user_lr', '--u_lr', type=float, default=0.01, help='user filter learning rate')
    parser.add_argument('--item_lr', '--i_lr', type=float, default=0.01, help='item filter learning rate')
    parser.add_argument('--bipartite_lr', '--b_lr', type=float, default=0.01, help='bipartite filter learning rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=5000, help='batch size for MSE training')
    parser.add_argument('--eval_freq', type=int, default=5, help='evaluate every N epochs')
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    
    # System
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--device', type=str, default='auto', help='device')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity')
    
    return parser.parse_args()


def get_config(args):
    """Convert args to config dict"""
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    return {
        'dataset': args.dataset,
        'u_eigen': args.u,
        'i_eigen': args.i, 
        'b_eigen': args.b,
        'use_laplacian': args.use_laplacian,
        'laplacian_type': args.laplacian_type,
        'user_lr': args.user_lr,
        'item_lr': args.item_lr,
        'bipartite_lr': args.bipartite_lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'eval_freq': args.eval_freq,
        'topks': eval(args.topks),
        'seed': args.seed,
        'device': device,
        'verbose': args.verbose
    }