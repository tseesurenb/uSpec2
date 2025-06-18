"""
Minimal config for v2 - only what we need
"""
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Raw Symmetric Softmax CF")
    
    parser.add_argument('--dataset', type=str, default='gowalla',
                       choices=['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book'])
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--temp_range', action='store_true', 
                       help='test range of temperatures instead of single value')
    
    return parser.parse_args()


def get_config(args):
    """Minimal config - only essentials plus required keys for dataloader"""
    return {
        # Essentials
        'dataset': args.dataset,
        'temperature': args.temperature,
        'temp_range': args.temp_range,
        
        # Required by dataloader (unused but needed to avoid errors)
        'topks': [20],
        'test_batch_size': 500,
        'train_batch_size': 1000,
        'verbose': 0,  # Turn off v1 verbose output
        'filter': 'i',
        'filter_type': 'bernstein',
        'filter_order': 8,
        'u_n_eigen': 25,
        'i_n_eigen': 200,
        'b_n_eigen': 220,
        'user_init': 'smooth',
        'item_init': 'sharp',
        'bipartite_init': 'smooth',
        'user_lr': 0.1,
        'item_lr': 0.01,
        'bipartite_lr': 0.05,
        'user_decay': 1e-4,
        'item_decay': 1e-3,
        'bipartite_decay': 5e-4,
        'epochs': 1,
        'neg_ratio': 4,
        'loss': 'mse',
        'optimizer': 'adam',
        'scheduler': 'none',
        'patience': 5,
        'eval_freq': 5,
        'seed': 2020,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'exp_name': 'v2',
        'save_model': False,
        'log_filters': False,
        'use_two_hop': False,
        'two_hop_weight': 0.3,
        'raw_only': False,
        'full_training': False,
    }