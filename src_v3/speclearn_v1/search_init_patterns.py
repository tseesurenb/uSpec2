#!/usr/bin/env python
"""
Search for optimal initialization patterns for spectral filters
Tests all combinations of init patterns for u, i, b views
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message="Can't initialize NVML")

import sys
import os
import itertools
import time
import pandas as pd
import subprocess
import json
from datetime import datetime

def run_experiment(dataset, u_init, i_init, b_init, base_args):
    """Run a single experiment with given init patterns"""
    
    # Build command
    cmd = [
        'python', 'main.py',
        '--dataset', dataset,
        '--user_init', u_init,
        '--item_init', i_init, 
        '--bipartite_init', b_init
    ]
    
    # Add base arguments
    for key, value in base_args.items():
        if key.startswith('--'):
            cmd.extend([key, str(value)])
        else:
            cmd.append(f'--{key}')
    
    # Run command and capture output
    try:
        print(f"Running: u={u_init}, i={i_init}, b={b_init}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        # Parse output for best NDCG
        best_ndcg = 0.0
        best_recall = 0.0
        best_precision = 0.0
        
        for line in result.stdout.split('\n'):
            if 'Best @' in line and 'NDCG' in line:
                # Extract NDCG value
                parts = line.split('NDCG:')[-1].strip()
                best_ndcg = float(parts.split()[0])
            elif 'Recall:' in line and best_ndcg > 0:
                # Get recall from same line
                parts = line.split('Recall:')[-1].strip()
                best_recall = float(parts.split()[0])
            elif 'Precision:' in line and best_ndcg > 0:
                # Get precision from same line
                parts = line.split('Precision:')[-1].strip()
                best_precision = float(parts.split()[0])
        
        return {
            'u_init': u_init,
            'i_init': i_init,
            'b_init': b_init,
            'ndcg': best_ndcg,
            'recall': best_recall,
            'precision': best_precision,
            'status': 'success',
            'error': None
        }
        
    except subprocess.TimeoutExpired:
        return {
            'u_init': u_init,
            'i_init': i_init,
            'b_init': b_init,
            'ndcg': 0.0,
            'recall': 0.0,
            'precision': 0.0,
            'status': 'timeout',
            'error': 'Timeout after 30 minutes'
        }
    except Exception as e:
        return {
            'u_init': u_init,
            'i_init': i_init,
            'b_init': b_init,
            'ndcg': 0.0,
            'recall': 0.0,
            'precision': 0.0,
            'status': 'error',
            'error': str(e)
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search init patterns")
    parser.add_argument('--dataset', type=str, default='yelp2018',
                       choices=['ml-100k', 'yelp2018', 'gowalla', 'amazon-book'])
    parser.add_argument('--quick', action='store_true',
                       help='Quick search with fewer patterns')
    
    # Allow overriding base parameters
    parser.add_argument('--u', type=int, default=160, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=500, help='item eigenvalues')
    parser.add_argument('--b', type=int, default=600, help='bipartite eigenvalues')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--user_lr', type=float, default=0.05, help='user learning rate')
    parser.add_argument('--item_lr', type=float, default=0.05, help='item learning rate')
    parser.add_argument('--bipartite_lr', type=float, default=0.05, help='bipartite learning rate')
    
    args = parser.parse_args()
    
    # Define base arguments for your experiment (using command line args)
    base_args = {
        '--full_training': '',
        '--u': str(args.u),
        '--i': str(args.i), 
        '--b': str(args.b),
        '--filter_type': 'spectral_basis',
        '--filter': 'uib',
        '--loss': 'mse',
        '--use_two_hop': '',
        '--epochs': str(args.epochs),
        '--user_lr': str(args.user_lr),
        '--item_lr': str(args.item_lr),
        '--bipartite_lr': str(args.bipartite_lr),
        '--eval_freq': '10'  # Evaluate less frequently for speed
    }
    
    # All available init patterns
    all_patterns = ['original', 'smooth', 'sharp', 'lowpass', 'uniform', 'lowfreq',
                    'linear_dec', 'step_0.5', 'step_0.7', 'step_0.9', 
                    'exp_decay', 'constant_1', 'constant_0.1']
    
    # For quick search, use a subset
    if args.quick:
        patterns = ['smooth', 'sharp', 'lowpass', 'linear_dec', 'step_0.7', 'exp_decay']
    else:
        patterns = all_patterns
    
    print(f"🔍 Init Pattern Search for {args.dataset}")
    print(f"Testing {len(patterns)} patterns per view")
    print(f"Total combinations: {len(patterns)**3}")
    print(f"Base config: {json.dumps(base_args, indent=2)}")
    
    # Generate all combinations
    combinations = list(itertools.product(patterns, patterns, patterns))
    
    # Run experiments
    results = []
    start_time = time.time()
    
    for i, (u_init, i_init, b_init) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing combination...")
        result = run_experiment(args.dataset, u_init, i_init, b_init, base_args)
        results.append(result)
        
        # Print current best
        if result['status'] == 'success' and result['ndcg'] > 0:
            print(f"  Result: NDCG={result['ndcg']:.4f}")
            
            # Show current top 3
            df_temp = pd.DataFrame(results)
            df_temp = df_temp[df_temp['status'] == 'success']
            if len(df_temp) > 0:
                df_sorted = df_temp.sort_values('ndcg', ascending=False)
                print("\n  Current Top 3:")
                for idx, row in df_sorted.head(3).iterrows():
                    print(f"    {row['u_init']}-{row['i_init']}-{row['b_init']}: {row['ndcg']:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Search completed in {elapsed/60:.1f} minutes")
    
    # Convert to DataFrame and analyze
    df = pd.DataFrame(results)
    df_success = df[df['status'] == 'success']
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"init_search_{args.dataset}_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"  Successful runs: {len(df_success)}/{len(df)}")
    print(f"  Failed runs: {len(df) - len(df_success)}")
    
    if len(df_success) > 0:
        # Sort by NDCG
        df_sorted = df_success.sort_values('ndcg', ascending=False)
        
        # Print top 10
        print(f"\n🏆 Top 10 Init Pattern Combinations:")
        print(f"{'Rank':<5} {'User':<12} {'Item':<12} {'Bipartite':<12} {'NDCG@20':<8} {'Recall@20':<10}")
        print("-" * 70)
        
        for idx, row in df_sorted.head(10).iterrows():
            rank = len(df_sorted) - df_sorted.index.get_loc(idx)
            print(f"{rank:<5} {row['u_init']:<12} {row['i_init']:<12} {row['b_init']:<12} "
                  f"{row['ndcg']:<8.4f} {row['recall']:<10.4f}")
        
        # Analyze patterns by view
        print("\n📈 Best Patterns by View:")
        
        # User view analysis
        u_avg = df_success.groupby('u_init')['ndcg'].agg(['mean', 'std', 'count'])
        u_avg = u_avg.sort_values('mean', ascending=False)
        print("\n  User View:")
        for pattern, stats in u_avg.head(5).iterrows():
            print(f"    {pattern:<12}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={int(stats['count'])})")
        
        # Item view analysis
        i_avg = df_success.groupby('i_init')['ndcg'].agg(['mean', 'std', 'count'])
        i_avg = i_avg.sort_values('mean', ascending=False)
        print("\n  Item View:")
        for pattern, stats in i_avg.head(5).iterrows():
            print(f"    {pattern:<12}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={int(stats['count'])})")
        
        # Bipartite view analysis
        b_avg = df_success.groupby('b_init')['ndcg'].agg(['mean', 'std', 'count'])
        b_avg = b_avg.sort_values('mean', ascending=False)
        print("\n  Bipartite View:")
        for pattern, stats in b_avg.head(5).iterrows():
            print(f"    {pattern:<12}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={int(stats['count'])})")
        
        # Best configuration
        best = df_sorted.iloc[0]
        print(f"\n🎉 Best Configuration:")
        print(f"   Init patterns: u={best['u_init']}, i={best['i_init']}, b={best['b_init']}")
        print(f"   NDCG@20: {best['ndcg']:.4f}")
        print(f"   Recall@20: {best['recall']:.4f}")
        print(f"   Precision@20: {best['precision']:.4f}")
        
        # Print command to reproduce
        print(f"\n🚀 Command to reproduce best result:")
        print(f"python main.py --dataset {args.dataset} {' '.join(f'{k} {v}' if v else k for k, v in base_args.items())} "
              f"--user_init {best['u_init']} --item_init {best['i_init']} --bipartite_init {best['b_init']}")