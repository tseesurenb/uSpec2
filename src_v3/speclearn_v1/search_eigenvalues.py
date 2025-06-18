#!/usr/bin/env python
"""
Eigenvalue Search for Learnable Spectral CF
Compute eigendecomposition once, then test different eigenvalue counts
Much faster than recomputing eigendecomposition for each combination
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
from main import Test

# Restore original sys.argv
sys.argv = original_argv

import argparse


class EigenSearchModel(SpectralCFLearnable):
    """Modified model that can use different eigen slices without recomputation"""
    
    def __init__(self, adj_mat, config, max_u, max_i, max_b):
        # Store max values
        self.max_u = max_u
        self.max_i = max_i  
        self.max_b = max_b
        
        # Temporarily set config to max values for initialization
        original_u = config.get('u_n_eigen', 25)
        original_i = config.get('i_n_eigen', 200)
        original_b = config.get('b_n_eigen', 220)
        
        config['u_n_eigen'] = max_u
        config['i_n_eigen'] = max_i
        config['b_n_eigen'] = max_b
        
        # Initialize with max eigenvalues
        super().__init__(adj_mat, config)
        
        # Restore original config
        config['u_n_eigen'] = original_u
        config['i_n_eigen'] = original_i
        config['b_n_eigen'] = original_b
        
        # Store full eigendecompositions
        self.full_user_eigenvals = self.user_eigenvals.clone() if hasattr(self, 'user_eigenvals') else None
        self.full_user_eigenvecs = self.user_eigenvecs.clone() if hasattr(self, 'user_eigenvecs') else None
        self.full_item_eigenvals = self.item_eigenvals.clone() if hasattr(self, 'item_eigenvals') else None
        self.full_item_eigenvecs = self.item_eigenvecs.clone() if hasattr(self, 'item_eigenvecs') else None
        self.full_bipartite_eigenvals = self.bipartite_eigenvals.clone() if hasattr(self, 'bipartite_eigenvals') else None
        self.full_bipartite_eigenvecs = self.bipartite_eigenvecs.clone() if hasattr(self, 'bipartite_eigenvecs') else None
    
    def set_eigen_slices(self, u_slice, i_slice, b_slice):
        """Set current eigenvalue slices for evaluation"""
        if self.full_user_eigenvals is not None and u_slice > 0:
            self.user_eigenvals = self.full_user_eigenvals[:u_slice]
            self.user_eigenvecs = self.full_user_eigenvecs[:, :u_slice]
        
        if self.full_item_eigenvals is not None and i_slice > 0:
            self.item_eigenvals = self.full_item_eigenvals[:i_slice]
            self.item_eigenvecs = self.full_item_eigenvecs[:, :i_slice]
            
        if self.full_bipartite_eigenvals is not None and b_slice > 0:
            self.bipartite_eigenvals = self.full_bipartite_eigenvals[:b_slice]
            self.bipartite_eigenvecs = self.full_bipartite_eigenvecs[:, :b_slice]


def evaluate_eigen_combination(model, dataset, u_eigen, i_eigen, b_eigen):
    """Evaluate a specific eigenvalue combination"""
    # Set eigenvalue slices
    model.set_eigen_slices(u_eigen, i_eigen, b_eigen)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        results = Test(dataset, model, 0)
    
    return {
        'u_eigen': u_eigen,
        'i_eigen': i_eigen, 
        'b_eigen': b_eigen,
        'ndcg': results['ndcg'][0],
        'recall': results['recall'][0],
        'precision': results['precision'][0]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eigenvalue Grid Search")
    parser.add_argument('--dataset', type=str, default='yelp2018',
                       choices=['ml-100k', 'yelp2018', 'gowalla', 'amazon-book'],
                       help='dataset to use')
    parser.add_argument('--filter', type=str, default='uib', help='filter views')
    parser.add_argument('--filter_type', type=str, default='spectral_basis',
                       choices=['bernstein', 'chebyshev', 'spectral_basis'],
                       help='filter type')
    parser.add_argument('--loss', type=str, default='mse', choices=['bpr', 'mse'],
                       help='loss function')
    
    # Eigenvalue ranges
    parser.add_argument('--u_min', type=int, default=10, help='min user eigenvalues')
    parser.add_argument('--u_max', type=int, default=150, help='max user eigenvalues')
    parser.add_argument('--i_min', type=int, default=50, help='min item eigenvalues')
    parser.add_argument('--i_max', type=int, default=300, help='max item eigenvalues')
    parser.add_argument('--b_min', type=int, default=50, help='min bipartite eigenvalues')
    parser.add_argument('--b_max', type=int, default=400, help='max bipartite eigenvalues')
    parser.add_argument('--step', type=int, default=25, help='step size')
    
    # Training parameters (for model initialization only)
    parser.add_argument('--user_lr', type=float, default=0.1)
    parser.add_argument('--item_lr', type=float, default=0.01)
    parser.add_argument('--bipartite_lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=1, help='training epochs (minimal for search)')
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'dataset': args.dataset,
        'full_training': True,
        'filter': args.filter,
        'filter_type': args.filter_type,
        'filter_order': 8,
        'user_lr': args.user_lr,
        'item_lr': args.item_lr,
        'bipartite_lr': args.bipartite_lr,
        'user_decay': 1e-4,
        'item_decay': 1e-3,
        'bipartite_decay': 5e-4,
        'epochs': args.epochs,
        'train_batch_size': 1000,
        'test_batch_size': 500,
        'neg_ratio': 4,
        'loss': args.loss,
        'optimizer': 'adam',
        'scheduler': 'none',
        'patience': 10,
        'topks': [20],
        'eval_freq': 1,
        'seed': 2020,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'verbose': 1,
        'exp_name': 'eigen_search',
        'save_model': False,
        'log_filters': False,
        'user_init': 'smooth',
        'item_init': 'sharp', 
        'bipartite_init': 'smooth'
    }
    
    print(f"🔍 Eigenvalue Grid Search")
    print(f"Dataset: {args.dataset}")
    print(f"Filter: {args.filter}, Type: {args.filter_type}")
    print(f"User range: {args.u_min}-{args.u_max}, Item range: {args.i_min}-{args.i_max}, Bipartite range: {args.b_min}-{args.b_max}")
    print(f"Step size: {args.step}")
    
    # Load dataset
    dataset = Loader(config)
    
    # Create model with maximum eigenvalues
    print(f"\n📊 Initializing model with max eigenvalues: u={args.u_max}, i={args.i_max}, b={args.b_max}")
    model = EigenSearchModel(dataset.UserItemNet, config, args.u_max, args.i_max, args.b_max).to(config['device'])
    
    # Generate eigenvalue combinations
    u_range = list(range(args.u_min, args.u_max + 1, args.step))
    i_range = list(range(args.i_min, args.i_max + 1, args.step))
    b_range = list(range(args.b_min, args.b_max + 1, args.step))
    
    # Filter based on active views
    if 'u' not in args.filter:
        u_range = [0]
    if 'i' not in args.filter:
        i_range = [0]
    if 'b' not in args.filter:
        b_range = [0]
    
    combinations = list(itertools.product(u_range, i_range, b_range))
    print(f"\n🎯 Testing {len(combinations)} combinations...")
    
    # Evaluate all combinations
    results = []
    start_time = time.time()
    
    for i, (u_eigen, i_eigen, b_eigen) in enumerate(combinations):
        print(f"\r[{i+1}/{len(combinations)}] Testing u={u_eigen}, i={i_eigen}, b={b_eigen}", end="", flush=True)
        
        result = evaluate_eigen_combination(model, dataset, u_eigen, i_eigen, b_eigen)
        results.append(result)
    
    elapsed = time.time() - start_time
    print(f"\n\n⏱️  Search completed in {elapsed:.1f}s ({elapsed/len(combinations):.2f}s per combination)")
    
    # Convert to DataFrame and sort by NDCG
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('ndcg', ascending=False)
    
    # Save results
    output_file = f"eigen_search_{args.dataset}_{args.filter}_{args.filter_type}_{args.loss}.csv"
    df_sorted.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")
    
    # Print top 10 results
    print(f"\n🏆 Top 10 Results:")
    print(f"{'Rank':<4} {'U':<4} {'I':<4} {'B':<4} {'NDCG@20':<8} {'Recall@20':<10} {'Precision@20':<12}")
    print("-" * 60)
    
    for idx, row in df_sorted.head(10).iterrows():
        print(f"{len(df_sorted) - df_sorted.index.get_loc(idx):<4} "
              f"{int(row['u_eigen']):<4} {int(row['i_eigen']):<4} {int(row['b_eigen']):<4} "
              f"{row['ndcg']:<8.4f} {row['recall']:<10.4f} {row['precision']:<12.4f}")
    
    # Print best result
    best = df_sorted.iloc[0]
    print(f"\n🎉 Best Configuration:")
    print(f"   Eigenvalues: u={int(best['u_eigen'])}, i={int(best['i_eigen'])}, b={int(best['b_eigen'])}")
    print(f"   NDCG@20: {best['ndcg']:.4f}")
    print(f"   Recall@20: {best['recall']:.4f}")
    print(f"   Precision@20: {best['precision']:.4f}")
    
    # Print command to reproduce best result
    print(f"\n🚀 Command to reproduce best result:")
    if args.loss == 'mse':
        print(f"python main.py --dataset {args.dataset} --filter {args.filter} "
              f"--filter_type {args.filter_type} --loss {args.loss} --full_training "
              f"--u {int(best['u_eigen'])} --i {int(best['i_eigen'])} --b {int(best['b_eigen'])} "
              f"--user_lr {args.user_lr} --item_lr {args.item_lr} --bipartite_lr {args.bipartite_lr}")
    else:
        print(f"python main.py --dataset {args.dataset} --filter {args.filter} "
              f"--filter_type {args.filter_type} --loss {args.loss} --full_training "
              f"--u {int(best['u_eigen'])} --i {int(best['i_eigen'])} --b {int(best['b_eigen'])} "
              f"--user_lr {args.user_lr} --item_lr {args.item_lr} --bipartite_lr {args.bipartite_lr}")


