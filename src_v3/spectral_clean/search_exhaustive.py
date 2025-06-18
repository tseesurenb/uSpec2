#!/usr/bin/env python3
"""
Exhaustive filter search - tests ALL combinations
Warning: This can be very slow! 14^3 = 2,744 combinations
"""

import numpy as np
import time
import json
from datetime import datetime
import itertools
import world
from dataloader import Loader, ML100K, LastFM
from model import SpectralCF
import Procedure
import argparse
from search_independent import EigenPrecomputedSpectralCF


def test_combination(model, dataset, user_filter, item_filter, bipartite_filter):
    """Test a specific filter combination"""
    model.user_filter_design = user_filter
    model.item_filter_design = item_filter
    model.bipartite_filter_design = bipartite_filter
    
    # Use optimal patterns
    model.user_init_filter = 'smooth'
    model.item_init_filter = 'sharp'
    model.bipartite_init_filter = 'smooth'
    
    try:
        start_time = time.time()
        results = Procedure.Test(dataset, model, 0, None, world.config['multicore'])
        test_time = time.time() - start_time
        
        return {
            'user_filter': user_filter,
            'item_filter': item_filter,
            'bipartite_filter': bipartite_filter,
            'precision': results['precision'][0],
            'recall': results['recall'][0],
            'ndcg': results['ndcg'][0],
            'test_time': test_time
        }
    except Exception as e:
        print(f"Error testing combination: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Exhaustive Filter Search")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'gowalla', 'yelp2018', 'amazon-book', 'lastfm'],
                       help='dataset to use')
    parser.add_argument('--u', type=int, default=25, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=200, help='item eigenvalues')
    parser.add_argument('--b', type=int, default=220, help='bipartite eigenvalues')
    parser.add_argument('--in_mat', type=str, default='uib',
                       help='input matrix: u, i, ui, b, ub, or uib')
    parser.add_argument('--max_combos', type=int, default=None,
                       help='limit number of combinations to test (for debugging)')
    parser.add_argument('--filters', nargs='+', default=None,
                       help='specific filters to test (default: all 14)')
    
    args = parser.parse_args()
    
    # Setup
    eigen_config = {
        'u_n_eigen': args.u,
        'i_n_eigen': args.i,
        'b_n_eigen': args.b,
        'in_mat': args.in_mat
    }
    
    print(f"EXHAUSTIVE SEARCH on {args.dataset}")
    print(f"Eigenvalues: u={args.u}, i={args.i}, b={args.b}")
    print(f"Active views: {args.in_mat}")
    
    # Load dataset
    if args.dataset == 'ml-100k':
        dataset = ML100K()
    elif world.dataset == 'lastfm':
        dataset = LastFM()
    elif args.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
        dataset = Loader(path=f"../../data/{args.dataset}")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")
    
    # Create base model and do eigendecomposition
    config = {**world.config, **eigen_config}
    base_model = SpectralCF(dataset.UserItemNet, config)
    
    print("\nPerforming eigendecomposition...")
    start_eigen = time.time()
    base_model.train()
    print(f"Eigendecomposition completed in {time.time() - start_eigen:.2f}s")
    
    # Create search model
    search_model = EigenPrecomputedSpectralCF(dataset.UserItemNet, config)
    search_model.set_precomputed_eigen(
        user_eigenvals=getattr(base_model, 'user_eigenvals', None),
        user_eigenvecs=getattr(base_model, 'user_eigenvecs', None),
        item_eigenvals=getattr(base_model, 'item_eigenvals', None),
        item_eigenvecs=getattr(base_model, 'item_eigenvecs', None),
        bipartite_eigenvals=getattr(base_model, 'bipartite_eigenvals', None),
        bipartite_eigenvecs=getattr(base_model, 'bipartite_eigenvecs', None)
    )
    
    # Filters to test
    if args.filters:
        filter_designs = args.filters
    else:
        filter_designs = ['original', 'chebyshev', 'jacobi', 'legendre', 'laguerre', 'hermite',
                         'bernstein', 'multiscale', 'bandstop', 'ensemble', 'golden', 'harmonic',
                         'spectral_basis', 'enhanced_basis']
    
    # Generate all combinations based on active views
    user_filters = filter_designs if 'u' in args.in_mat else ['original']
    item_filters = filter_designs if 'i' in args.in_mat else ['original']
    bipartite_filters = filter_designs if 'b' in args.in_mat else ['original']
    
    all_combinations = list(itertools.product(user_filters, item_filters, bipartite_filters))
    
    if args.max_combos:
        all_combinations = all_combinations[:args.max_combos]
    
    print(f"\nTesting {len(all_combinations)} combinations...")
    print(f"This will take approximately {len(all_combinations) * 0.5:.1f} seconds")
    
    results = []
    best_result = None
    
    # Progress tracking
    start_time = time.time()
    
    for i, (user_filter, item_filter, bipartite_filter) in enumerate(all_combinations):
        # Progress update every 10%
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
    
    for view in ['user_filter', 'item_filter', 'bipartite_filter']:
        filter_counts = {}
        for result in top_20:
            filter_name = result[view]
            filter_counts[filter_name] = filter_counts.get(filter_name, 0) + 1
        
        print(f"\n{view.replace('_filter', '').capitalize()} - most common in top 20:")
        for filter_name, count in sorted(filter_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {filter_name}: {count}/20 ({count/20*100:.0f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"exhaustive_search_{args.dataset}_{timestamp}.json"
    
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
              f"--uf {best_result['user_filter']} --if {best_result['item_filter']} "
              f"--bf {best_result['bipartite_filter']}")


if __name__ == "__main__":
    main()