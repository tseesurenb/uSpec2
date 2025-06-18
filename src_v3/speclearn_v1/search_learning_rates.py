#!/usr/bin/env python
"""
Learning Rate Search for Learnable Spectral CF
Independent search for optimal learning rates and weight decay per view
"""

import sys
import os

# Temporarily override sys.argv to prevent world.py from parsing our args
original_argv = sys.argv
sys.argv = [sys.argv[0]]  # Keep only script name

import torch
import numpy as np
import time
import itertools
import pandas as pd

# Only import these after setting sys.argv
from learnable_model import SpectralCFLearnable
from dataloader import Loader
from main import Test, get_optimizer, MSE_train_learnable, BPR_train_learnable

# Restore original sys.argv
sys.argv = original_argv

import argparse


def train_and_evaluate(dataset, config, epochs=30):
    """Train model and return best NDCG"""
    # Create model
    model = SpectralCFLearnable(dataset.UserItemNet, config).to(config['device'])
    
    # Create optimizer with per-view hyperparameters
    optimizer = get_optimizer(model, config)
    
    # Training loop
    best_ndcg = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        if config['loss'] == 'bpr':
            train_loss = BPR_train_learnable(
                dataset, model, optimizer, 
                neg_ratio=config['neg_ratio'],
                batch_size=config['train_batch_size']
            )
        else:  # MSE
            train_loss = MSE_train_learnable(
                dataset, model, optimizer,
                batch_size=config['train_batch_size']
            )
        
        # Evaluation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                results = Test(dataset, model, epoch)
            
            ndcg = results['ndcg'][0]
            if ndcg > best_ndcg:
                best_ndcg = ndcg
    
    return best_ndcg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning Rate and Decay Search")
    
    # Dataset and model
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'yelp2018', 'gowalla', 'amazon-book'])
    parser.add_argument('--filter', type=str, default='uib', help='filter views')
    parser.add_argument('--filter_type', type=str, default='spectral_basis',
                       choices=['bernstein', 'chebyshev', 'spectral_basis'])
    parser.add_argument('--loss', type=str, default='mse', choices=['bpr', 'mse'])
    
    # Fixed eigenvalues
    parser.add_argument('--u', type=int, default=10, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=300, help='item eigenvalues')
    parser.add_argument('--b', type=int, default=50, help='bipartite eigenvalues')
    
    parser.add_argument('--epochs', type=int, default=30, help='training epochs per config')
    
    args = parser.parse_args()
    
    print(f"🔍 Learning Rate and Decay Grid Search")
    print(f"Dataset: {args.dataset}")
    print(f"Filter: {args.filter}, Type: {args.filter_type}, Loss: {args.loss}")
    print(f"Fixed eigenvalues: u={args.u}, i={args.i}, b={args.b}")
    
    # Expanded search space for each view (more options since independent search)
    user_lrs = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]  # User view typically needs higher LR
    item_lrs = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]  # Item view typically needs lower LR  
    bipartite_lrs = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]  # Bipartite similar to user
    
    user_decays = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5]
    item_decays = [1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]  # Item might need stronger regularization
    bipartite_decays = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5]
    
    # Base config
    base_config = {
        'dataset': args.dataset,
        'full_training': True,
        'filter': args.filter,
        'filter_type': args.filter_type,
        'filter_order': 8,
        'u_n_eigen': args.u,
        'i_n_eigen': args.i,
        'b_n_eigen': args.b,
        'epochs': args.epochs,
        'train_batch_size': 1000,
        'test_batch_size': 500,
        'neg_ratio': 4,
        'loss': args.loss,
        'optimizer': 'adam',
        'scheduler': 'none',
        'patience': 10,
        'topks': [20],
        'eval_freq': 5,
        'seed': 2020,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'verbose': 0,
        'exp_name': 'lr_search',
        'save_model': False,
        'log_filters': False,
        'user_init': 'smooth',
        'item_init': 'sharp',
        'bipartite_init': 'smooth'
    }
    
    # Load dataset once
    dataset = Loader(base_config)
    
    print(f"\n🎯 Independent view search (much faster!)...")
    print(f"User LRs: {user_lrs}, Decays: {user_decays}")
    print(f"Item LRs: {item_lrs}, Decays: {item_decays}")
    print(f"Bipartite LRs: {bipartite_lrs}, Decays: {bipartite_decays}")
    
    # Search each view independently
    results = []
    start_time = time.time()
    
    # Default values for other views
    default_config = {
        'user_lr': 0.01,
        'item_lr': 0.001, 
        'bipartite_lr': 0.01,
        'user_decay': 1e-4,
        'item_decay': 1e-3,
        'bipartite_decay': 1e-4
    }
    
    test_count = 0
    total_tests = len(user_lrs) * len(user_decays) + len(item_lrs) * len(item_decays) + len(bipartite_lrs) * len(bipartite_decays)
    
    # 1. Search user view
    print(f"\n🔍 Searching user view...")
    for u_lr in user_lrs:
        for u_d in user_decays:
            test_count += 1
            print(f"\r[{test_count}/{total_tests}] Testing user: lr={u_lr:.1e}, decay={u_d:.1e}", end="", flush=True)
            
            config = base_config.copy()
            config.update(default_config)
            config.update({'user_lr': u_lr, 'user_decay': u_d})
            
            try:
                ndcg = train_and_evaluate(dataset, config, args.epochs)
                results.append({
                    'view': 'user',
                    'user_lr': u_lr, 'item_lr': default_config['item_lr'], 'bipartite_lr': default_config['bipartite_lr'],
                    'user_decay': u_d, 'item_decay': default_config['item_decay'], 'bipartite_decay': default_config['bipartite_decay'],
                    'ndcg': ndcg
                })
            except Exception as e:
                print(f"\nError: {e}")
                continue
    
    # 2. Search item view  
    print(f"\n🔍 Searching item view...")
    for i_lr in item_lrs:
        for i_d in item_decays:
            test_count += 1
            print(f"\r[{test_count}/{total_tests}] Testing item: lr={i_lr:.1e}, decay={i_d:.1e}", end="", flush=True)
            
            config = base_config.copy()
            config.update(default_config)
            config.update({'item_lr': i_lr, 'item_decay': i_d})
            
            try:
                ndcg = train_and_evaluate(dataset, config, args.epochs)
                results.append({
                    'view': 'item',
                    'user_lr': default_config['user_lr'], 'item_lr': i_lr, 'bipartite_lr': default_config['bipartite_lr'],
                    'user_decay': default_config['user_decay'], 'item_decay': i_d, 'bipartite_decay': default_config['bipartite_decay'],
                    'ndcg': ndcg
                })
            except Exception as e:
                print(f"\nError: {e}")
                continue
    
    # 3. Search bipartite view
    print(f"\n🔍 Searching bipartite view...")
    for b_lr in bipartite_lrs:
        for b_d in bipartite_decays:
            test_count += 1
            print(f"\r[{test_count}/{total_tests}] Testing bipartite: lr={b_lr:.1e}, decay={b_d:.1e}", end="", flush=True)
            
            config = base_config.copy()
            config.update(default_config)
            config.update({'bipartite_lr': b_lr, 'bipartite_decay': b_d})
            
            try:
                ndcg = train_and_evaluate(dataset, config, args.epochs)
                results.append({
                    'view': 'bipartite',
                    'user_lr': default_config['user_lr'], 'item_lr': default_config['item_lr'], 'bipartite_lr': b_lr,
                    'user_decay': default_config['user_decay'], 'item_decay': default_config['item_decay'], 'bipartite_decay': b_d,
                    'ndcg': ndcg
                })
            except Exception as e:
                print(f"\nError: {e}")
                continue
    
    elapsed = time.time() - start_time
    print(f"\n\n⏱️  Search completed in {elapsed/60:.1f} minutes")
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('ndcg', ascending=False)
    
    # Save results
    output_file = f"lr_search_{args.dataset}_{args.filter}_{args.filter_type}_{args.loss}_u{args.u}_i{args.i}_b{args.b}_independent.csv"
    df_sorted.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")
    
    # Print best results per view
    print(f"\n🏆 Best Results per View:")
    for view in ['user', 'item', 'bipartite']:
        view_results = df[df['view'] == view].sort_values('ndcg', ascending=False)
        if len(view_results) > 0:
            best_view = view_results.iloc[0]
            print(f"\n{view.capitalize()} view:")
            print(f"  Best LR: {best_view[f'{view}_lr']:.1e}, Decay: {best_view[f'{view}_decay']:.1e}")
            print(f"  NDCG@20: {best_view['ndcg']:.4f}")
    
    # Print overall best
    best = df_sorted.iloc[0]
    print(f"\n🎉 Overall Best Configuration:")
    print(f"   View: {best['view']}")
    print(f"   Learning Rates: user={best['user_lr']:.1e}, item={best['item_lr']:.1e}, bipartite={best['bipartite_lr']:.1e}")
    print(f"   Weight Decays: user={best['user_decay']:.1e}, item={best['item_decay']:.1e}, bipartite={best['bipartite_decay']:.1e}")
    print(f"   NDCG@20: {best['ndcg']:.4f}")
    
    # Print command to reproduce
    print(f"\n🚀 Command to reproduce best result:")
    print(f"python main.py --dataset {args.dataset} --filter {args.filter} "
          f"--filter_type {args.filter_type} --loss {args.loss} --full_training "
          f"--u {args.u} --i {args.i} --b {args.b} "
          f"--user_lr {best['user_lr']:.1e} --item_lr {best['item_lr']:.1e} --bipartite_lr {best['bipartite_lr']:.1e} "
          f"--user_decay {best['user_decay']:.1e} --item_decay {best['item_decay']:.1e} --bipartite_decay {best['bipartite_decay']:.1e} "
          f"--epochs 80")