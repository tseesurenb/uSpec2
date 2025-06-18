#!/usr/bin/env python3
"""
Exhaustive filter search for learnable model
Tests ALL possible combinations
"""

import numpy as np
import time
import json
from datetime import datetime
import itertools
import torch
import world
import utils
from dataloader import Loader
from model_simplified import SimplifiedSpectralCF
import procedure
import argparse
from search_greedy import PrecomputedSpectralCF


def test_combination(model, dataset, user_filter, item_filter, bipartite_filter):
    """Test a specific filter combination"""
    # Map short names to full names
    model.user_filter_design = model.filter_mapping.get(user_filter, user_filter)
    model.item_filter_design = model.filter_mapping.get(item_filter, item_filter)
    model.bipartite_filter_design = model.filter_mapping.get(bipartite_filter, bipartite_filter)
    
    # Use optimal patterns
    model.user_init_filter = 'smooth'
    model.item_init_filter = 'sharp'
    model.bipartite_init_filter = 'smooth'
    
    try:
        start_time = time.time()
        results = procedure.Test(dataset, model, 0)
        test_time = time.time() - start_time
        
        return {
            'user_filter': user_filter,
            'item_filter': item_filter,
            'bipartite_filter': bipartite_filter,
            'precision': float(results['precision'][0]),
            'recall': float(results['recall'][0]),
            'ndcg': float(results['ndcg'][0]),
            'test_time': test_time
        }
    except Exception as e:
        print(f"Error testing combination: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Exhaustive Filter Search - Learnable Model")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'gowalla', 'yelp2018', 'amazon-book'],
                       help='dataset to use')
    parser.add_argument('--u', type=int, default=8, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=40, help='item eigenvalues')
    parser.add_argument('--b', type=int, default=60, help='bipartite eigenvalues')
    parser.add_argument('--in_mat', type=str, default='uib',
                       help='which views to include')
    parser.add_argument('--max_combos', type=int, default=None,
                       help='limit number of combinations to test')
    parser.add_argument('--filters', nargs='+', default=None,
                       help='specific filters to test (default: all 14)')
    
    args = parser.parse_args()
    
    # Import necessary modules
    import os
    import sys
    import torch.nn as nn
    import scipy.sparse as sp
    
    # Update world config
    world.config['u_n_eigen'] = args.u
    world.config['i_n_eigen'] = args.i
    world.config['b_n_eigen'] = args.b
    world.config['filter'] = args.in_mat
    world.config['dataset'] = args.dataset
    
    print(f"EXHAUSTIVE SEARCH - Learnable Model")
    print(f"Dataset: {args.dataset}")
    print(f"Eigenvalues: u={args.u}, i={args.i}, b={args.b}")
    print(f"Active views: {args.in_mat}")
    
    # Load dataset
    dataset = Loader(args.dataset)
    
    # Create base model and do eigendecomposition
    print("\nPerforming eigendecomposition...")
    start_eigen = time.time()
    base_model = SimplifiedSpectralCF(dataset.UserItemNet, world.config)
    print(f"Eigendecomposition completed in {time.time() - start_eigen:.2f}s")
    
    # Create search model that reuses eigendecompositions
    search_model = PrecomputedSpectralCF(dataset.UserItemNet, world.config)
    
    # Copy eigendecompositions based on active views
    if 'u' in args.in_mat and hasattr(base_model, 'user_eigenvals'):
        search_model.set_precomputed_eigen(
            user_eigenvals=base_model.user_eigenvals,
            user_eigenvecs=base_model.user_eigenvecs
        )
    if 'i' in args.in_mat and hasattr(base_model, 'item_eigenvals'):
        search_model.set_precomputed_eigen(
            item_eigenvals=base_model.item_eigenvals,
            item_eigenvecs=base_model.item_eigenvecs
        )
    if 'b' in args.in_mat and hasattr(base_model, 'bipartite_eigenvals'):
        search_model.set_precomputed_eigen(
            bipartite_eigenvals=base_model.bipartite_eigenvals,
            bipartite_eigenvecs=base_model.bipartite_eigenvecs
        )
    
    # Filters to test (short names)
    if args.filters:
        filter_designs = args.filters
    else:
        filter_designs = ['orig', 'cheby', 'jacobi', 'legendre', 'laguerre', 'hermite',
                         'bernstein', 'multi', 'band', 'ensemble', 'golden', 'harmonic',
                         'spectral_basis', 'enhanced_basis']
    
    # Generate all combinations based on active views
    user_filters = filter_designs if 'u' in args.in_mat else ['orig']
    item_filters = filter_designs if 'i' in args.in_mat else ['orig']
    bipartite_filters = filter_designs if 'b' in args.in_mat else ['orig']
    
    all_combinations = list(itertools.product(user_filters, item_filters, bipartite_filters))
    
    if args.max_combos:
        all_combinations = all_combinations[:args.max_combos]
    
    print(f"\nTesting {len(all_combinations)} combinations...")
    print(f"Estimated time: {len(all_combinations) * 0.5:.1f} seconds")
    
    results = []
    best_result = None
    
    # Progress tracking
    start_time = time.time()
    
    for i, (user_filter, item_filter, bipartite_filter) in enumerate(all_combinations):
        # Progress update
        if i % max(1, len(all_combinations) // 10) == 0:
            elapsed = time.time() - start_time
            progress = (i / len(all_combinations)) * 100
            eta = (elapsed / (i + 1)) * (len(all_combinations) - i - 1) if i > 0 else 0
            print(f"\nProgress: {i}/{len(all_combinations)} ({progress:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")
            if best_result:
                print(f"Current best: u={best_result['user_filter']}, "
                      f"i={best_result['item_filter']}, b={best_result['bipartite_filter']} "
                      f"(NDCG: {best_result['ndcg']:.4f})")
        
        # Test combination
        result = test_combination(search_model, dataset, user_filter, item_filter, bipartite_filter)
        
        if result:
            results.append(result)
            
            # Update best
            if best_result is None or result['ndcg'] > best_result['ndcg']:
                best_result = result.copy()
                print(f"\n✓ New best! u={user_filter}, i={item_filter}, "
                      f"b={bipartite_filter} → NDCG: {result['ndcg']:.4f}")
    
    # Sort results by NDCG
    results.sort(key=lambda x: x['ndcg'], reverse=True)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n\n{'='*60}")
    print("EXHAUSTIVE SEARCH COMPLETED")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Combinations tested: {len(results)}")
    print(f"Average time per test: {total_time/len(results):.2f} seconds")
    
    print(f"\n\nTop 10 combinations:")
    for i, result in enumerate(results[:10]):
        print(f"\n{i+1}. NDCG: {result['ndcg']:.4f} | Recall: {result['recall']:.4f} | "
              f"Precision: {result['precision']:.4f}")
        print(f"   User: {result['user_filter']}")
        print(f"   Item: {result['item_filter']}")
        print(f"   Bipartite: {result['bipartite_filter']}")
    
    # Analysis
    print("\n\nAnalysis of top 20 combinations:")
    top_20 = results[:20]
    
    for view_key in ['user_filter', 'item_filter', 'bipartite_filter']:
        filter_counts = {}
        for result in top_20:
            filter_name = result[view_key]
            filter_counts[filter_name] = filter_counts.get(filter_name, 0) + 1
        
        view_name = view_key.replace('_filter', '').capitalize()
        print(f"\n{view_name} - most common in top 20:")
        for filter_name, count in sorted(filter_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {filter_name}: {count}/20 ({count/20*100:.0f}%)")
    
    # Find interaction patterns
    print("\n\nCommon patterns in top 10:")
    pattern_counts = {}
    for result in results[:10]:
        pattern = f"{result['user_filter']}-{result['item_filter']}-{result['bipartite_filter']}"
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"exhaustive_search_learnable_{args.dataset}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'eigenvalues': {'u': args.u, 'i': args.i, 'b': args.b},
            'in_mat': args.in_mat,
            'filters_tested': filter_designs,
            'total_combinations': len(results),
            'total_time_seconds': total_time,
            'top_10_results': results[:10],
            'all_results': results,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n\nFull results saved to {results_file}")
    
    # Print command for best combination
    if best_result:
        print(f"\nTo run the best configuration:")
        print(f"python main.py --dataset {args.dataset} --in_mat {args.in_mat} "
              f"--u {args.u} --i {args.i} --b {args.b} "
              f"--uf {best_result['user_filter']} --if_ {best_result['item_filter']} "
              f"--bf {best_result['bipartite_filter']}")


if __name__ == "__main__":
    main()