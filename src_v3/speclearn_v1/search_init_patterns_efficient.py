#!/usr/bin/env python
"""
Efficient Init Pattern Search for Learnable Spectral CF
Loads data and computes eigendecomposition only once
Only changes filter initialization between experiments
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message="Can't initialize NVML")

import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
from datetime import datetime
import os
import copy

# Import from current directory
from config import get_config
from learnable_model import SpectralCFLearnable
from dataloader import Loader
from main import get_optimizer, Test, MSE_train_learnable, BPR_train_learnable
import utils


class InitPatternSearcher:
    """Efficient searcher that reuses eigendecomposition"""
    
    def __init__(self, dataset_name, config):
        self.dataset_name = dataset_name
        self.config = config
        
        print(f"📊 Loading dataset {dataset_name}...")
        self.dataset = Loader(config)
        
        print(f"🔧 Creating base model...")
        # Create model with default initialization
        self.base_model = SpectralCFLearnable(self.dataset.UserItemNet, config).to(config['device'])
        
        # Store the eigendecompositions and base tensors
        self.store_base_state()
        
    def store_base_state(self):
        """Store eigendecompositions and other reusable components"""
        self.base_state = {}
        
        # Store eigendecompositions
        if hasattr(self.base_model, 'user_eigenvals'):
            self.base_state['user_eigenvals'] = self.base_model.user_eigenvals.clone()
            self.base_state['user_eigenvecs'] = self.base_model.user_eigenvecs.clone()
            
        if hasattr(self.base_model, 'item_eigenvals'):
            self.base_state['item_eigenvals'] = self.base_model.item_eigenvals.clone()
            self.base_state['item_eigenvecs'] = self.base_model.item_eigenvecs.clone()
            
        if hasattr(self.base_model, 'bipartite_eigenvals'):
            self.base_state['bipartite_eigenvals'] = self.base_model.bipartite_eigenvals.clone()
            self.base_state['bipartite_eigenvecs'] = self.base_model.bipartite_eigenvecs.clone()
            
        # Store other tensors
        self.base_state['adj_tensor'] = self.base_model.adj_tensor.clone()
        if hasattr(self.base_model, 'two_hop_matrix'):
            self.base_state['two_hop_matrix'] = self.base_model.two_hop_matrix.clone()
    
    def create_model_with_init(self, u_init, i_init, b_init):
        """Create a new model with specific initialization patterns"""
        # Update config with new init patterns
        new_config = self.config.copy()
        new_config['user_init'] = u_init
        new_config['item_init'] = i_init
        new_config['bipartite_init'] = b_init
        
        # Create new model
        model = SpectralCFLearnable(self.dataset.UserItemNet, new_config).to(self.config['device'])
        
        # Copy eigendecompositions from base model (skip recomputation!)
        with torch.no_grad():
            for key, value in self.base_state.items():
                if hasattr(model, key):
                    setattr(model, key, value.clone())
        
        # The filters are already initialized with the new patterns
        # No need to recompute eigendecomposition
        
        return model
    
    def evaluate_init_pattern(self, u_init, i_init, b_init):
        """Evaluate a specific init pattern combination"""
        print(f"\nTesting: u={u_init}, i={i_init}, b={b_init}")
        
        # Create model with specific initialization
        model = self.create_model_with_init(u_init, i_init, b_init)
        
        # Create optimizer
        optimizer = get_optimizer(model, self.config)
        
        # Training
        best_ndcg = 0.0
        best_recall = 0.0
        best_precision = 0.0
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Train
            model.train()
            if self.config['loss'] == 'mse':
                mse_loss = MSE_train_learnable(self.dataset, model, optimizer)
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch+1}: MSE Loss = {mse_loss:.4f}")
            else:
                bpr_loss = BPR_train_learnable(self.dataset, model, optimizer)
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch+1}: BPR Loss = {bpr_loss:.4f}")
            
            # Evaluate
            if (epoch + 1) % self.config['eval_freq'] == 0:
                model.eval()
                with torch.no_grad():
                    results = Test(self.dataset, model, epoch)
                    ndcg = results['ndcg'][0]
                    recall = results['recall'][0]
                    precision = results['precision'][0]
                    
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        best_recall = recall
                        best_precision = precision
                    
                    if epoch % 10 == 0:
                        print(f"  Test NDCG@20: {ndcg:.4f} | Recall@20: {recall:.4f} | Precision@20: {precision:.4f}")
        
        elapsed = time.time() - start_time
        print(f"  Training time: {elapsed:.1f}s")
        print(f"  Best NDCG@20: {best_ndcg:.4f}")
        
        return {
            'u_init': u_init,
            'i_init': i_init,
            'b_init': b_init,
            'ndcg': best_ndcg,
            'recall': best_recall,
            'precision': best_precision,
            'time': elapsed
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Efficient Init Pattern Search")
    parser.add_argument('--dataset', type=str, default='yelp2018',
                       choices=['ml-100k', 'lastfm', 'yelp2018', 'gowalla', 'amazon-book'])
    parser.add_argument('--quick', action='store_true',
                       help='Quick search with fewer patterns')
    
    # Model parameters
    parser.add_argument('--u', type=int, default=160, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=500, help='item eigenvalues')
    parser.add_argument('--b', type=int, default=600, help='bipartite eigenvalues')
    parser.add_argument('--filter_type', type=str, default='spectral_basis')
    parser.add_argument('--filter', type=str, default='uib')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'bpr'])
    parser.add_argument('--use_two_hop', action='store_true')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--user_lr', type=float, default=0.05)
    parser.add_argument('--item_lr', type=float, default=0.05)
    parser.add_argument('--bipartite_lr', type=float, default=0.05)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--train_batch', type=int, default=1000)
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'dataset': args.dataset,
        'full_training': True,
        'filter': args.filter,
        'filter_type': args.filter_type,
        'filter_order': 8,
        'u_n_eigen': args.u,
        'i_n_eigen': args.i,
        'b_n_eigen': args.b,
        'user_lr': args.user_lr,
        'item_lr': args.item_lr,
        'bipartite_lr': args.bipartite_lr,
        'user_decay': 1e-3,
        'item_decay': 1e-3,
        'bipartite_decay': 1e-3,
        'epochs': args.epochs,
        'train_batch_size': args.train_batch,
        'test_batch_size': 500,
        'neg_ratio': 4,
        'loss': args.loss,
        'optimizer': 'adam',
        'scheduler': 'none',
        'patience': 10,
        'topks': [20],
        'eval_freq': args.eval_freq,
        'seed': 2020,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'verbose': 1,
        'exp_name': 'init_search',
        'save_model': False,
        'log_filters': False,
        'use_two_hop': args.use_two_hop,
        'two_hop_weight': 1.3,
        'raw_only': False
    }
    
    # Set random seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # All available init patterns
    all_patterns = ['original', 'smooth', 'sharp', 'lowpass', 'uniform', 'lowfreq',
                    'linear_dec', 'step_0.5', 'step_0.7', 'step_0.9', 
                    'exp_decay', 'constant_1', 'constant_0.1']
    
    # For quick search, use a subset
    if args.quick:
        patterns = ['smooth', 'sharp', 'lowpass', 'linear_dec', 'step_0.7', 'exp_decay']
    else:
        patterns = all_patterns
    
    print(f"🔍 Efficient Init Pattern Search")
    print(f"Dataset: {args.dataset}")
    print(f"Testing {len(patterns)} patterns per view = {len(patterns)**3} total combinations")
    print(f"Model: {args.filter} views, {args.filter_type} filters")
    print(f"Eigenvalues: u={args.u}, i={args.i}, b={args.b}")
    print(f"Training: {args.epochs} epochs, lr={args.user_lr}/{args.item_lr}/{args.bipartite_lr}")
    
    # Create searcher (loads data and computes eigendecomposition once)
    start_time = time.time()
    searcher = InitPatternSearcher(args.dataset, config)
    print(f"✅ Setup completed in {time.time() - start_time:.1f}s")
    
    # Test all combinations
    results = []
    best_ndcg = 0.0
    best_config = None
    
    import itertools
    combinations = list(itertools.product(patterns, patterns, patterns))
    
    for i, (u_init, i_init, b_init) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] ", end='')
        
        result = searcher.evaluate_init_pattern(u_init, i_init, b_init)
        results.append(result)
        
        # Track best
        if result['ndcg'] > best_ndcg:
            best_ndcg = result['ndcg']
            best_config = (u_init, i_init, b_init)
        
        # Show comparison
        if result['ndcg'] > 0:
            print(f"  Current: NDCG={result['ndcg']:.4f}")
            if best_config:
                print(f"  Best so far: NDCG={best_ndcg:.4f} (u={best_config[0]}, i={best_config[1]}, b={best_config[2]})")
    
    # Save and analyze results
    total_time = sum(r['time'] for r in results)
    print(f"\n⏱️  Total search time: {total_time/60:.1f} minutes")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"init_search_{args.dataset}_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Sort by NDCG
    df_sorted = df.sort_values('ndcg', ascending=False)
    
    # Print top 10
    print(f"\n🏆 Top 10 Init Pattern Combinations:")
    print(f"{'Rank':<5} {'User':<12} {'Item':<12} {'Bipartite':<12} {'NDCG@20':<8} {'Time(s)':<8}")
    print("-" * 65)
    
    for idx, row in df_sorted.head(10).iterrows():
        rank = idx + 1
        print(f"{rank:<5} {row['u_init']:<12} {row['i_init']:<12} {row['b_init']:<12} "
              f"{row['ndcg']:<8.4f} {row['time']:<8.1f}")
    
    # Analyze by view
    print("\n📈 Best Patterns by View:")
    
    for view_name, col_name in [('User', 'u_init'), ('Item', 'i_init'), ('Bipartite', 'b_init')]:
        view_stats = df.groupby(col_name)['ndcg'].agg(['mean', 'std', 'count', 'max'])
        view_stats = view_stats.sort_values('mean', ascending=False)
        
        print(f"\n  {view_name} View:")
        for pattern, stats in view_stats.head(5).iterrows():
            print(f"    {pattern:<12}: mean={stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"(max={stats['max']:.4f}, n={int(stats['count'])})")
    
    # Best configuration
    best = df_sorted.iloc[0]
    print(f"\n🎉 Best Configuration:")
    print(f"   Init patterns: u={best['u_init']}, i={best['i_init']}, b={best['b_init']}")
    print(f"   NDCG@20: {best['ndcg']:.4f}")
    print(f"   Recall@20: {best['recall']:.4f}")
    print(f"   Precision@20: {best['precision']:.4f}")
    print(f"   Training time: {best['time']:.1f}s")
    
    # Command to reproduce
    print(f"\n🚀 Command to reproduce best result:")
    print(f"python main.py --dataset {args.dataset} --full_training "
          f"--filter {args.filter} --filter_type {args.filter_type} --loss {args.loss} "
          f"--u {args.u} --i {args.i} --b {args.b} "
          f"--user_init {best['u_init']} --item_init {best['i_init']} --bipartite_init {best['b_init']} "
          f"--user_lr {args.user_lr} --item_lr {args.item_lr} --bipartite_lr {args.bipartite_lr} "
          f"--epochs {args.epochs}", end='')
    if args.use_two_hop:
        print(" --use_two_hop", end='')
    print()


if __name__ == "__main__":
    main()