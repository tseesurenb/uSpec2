#!/usr/bin/env python3
"""
Interaction-aware filter search
Tests promising combinations based on interaction patterns
"""

import numpy as np
import time
import json
from datetime import datetime
import itertools
import world
from dataloader import Loader, ML100K
from model import SpectralCF
import Procedure
import argparse
from search_independent import EigenPrecomputedSpectralCF


# Known good interactions from empirical observations
INTERACTION_PATTERNS = {
    'complementary': [
        # Filters that work well together
        ('multiscale', 'chebyshev', 'original'),
        ('enhanced_basis', 'laguerre', 'original'),
        ('spectral_basis', 'hermite', 'bandstop'),
        ('original', 'golden', 'harmonic'),
    ],
    'similar': [
        # Similar filters that might work well
        ('chebyshev', 'jacobi', 'legendre'),
        ('multiscale', 'enhanced_basis', 'spectral_basis'),
        ('bandstop', 'harmonic', 'golden'),
    ],
    'diverse': [
        # Diverse filters for different views
        ('original', 'ensemble', 'bandstop'),
        ('hermite', 'multiscale', 'original'),
        ('laguerre', 'golden', 'chebyshev'),
    ]
}


def generate_smart_combinations():
    """Generate combinations based on interaction patterns"""
    combinations = []
    
    # Add all predefined patterns
    for pattern_type, patterns in INTERACTION_PATTERNS.items():
        for combo in patterns:
            combinations.append({
                'user': combo[0],
                'item': combo[1],
                'bipartite': combo[2],
                'pattern': pattern_type
            })
    
    # Add variations (permutations of good combinations)
    for pattern_type, patterns in INTERACTION_PATTERNS.items():
        for combo in patterns[:3]:  # Only permute top 3 from each category
            for perm in itertools.permutations(combo):
                combo_dict = {
                    'user': perm[0],
                    'item': perm[1],
                    'bipartite': perm[2],
                    'pattern': f'{pattern_type}_permuted'
                }
                if combo_dict not in combinations:
                    combinations.append(combo_dict)
    
    return combinations


def test_combination(model, dataset, combo):
    """Test a specific filter combination"""
    model.user_filter_design = combo['user']
    model.item_filter_design = combo['item']
    model.bipartite_filter_design = combo['bipartite']
    
    # Use optimal patterns
    model.user_init_filter = 'smooth'
    model.item_init_filter = 'sharp'
    model.bipartite_init_filter = 'smooth'
    
    try:
        start_time = time.time()
        results = Procedure.Test(dataset, model, 0, None, world.config['multicore'])
        test_time = time.time() - start_time
        
        return {
            **combo,
            'precision': results['precision'][0],
            'recall': results['recall'][0],
            'ndcg': results['ndcg'][0],
            'test_time': test_time
        }
    except Exception as e:
        print(f"Error testing combination: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Interaction-Aware Filter Search")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'gowalla', 'yelp2018', 'amazon-book'],
                       help='dataset to use')
    parser.add_argument('--u', type=int, default=25, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=200, help='item eigenvalues')
    parser.add_argument('--b', type=int, default=220, help='bipartite eigenvalues')
    parser.add_argument('--top_k', type=int, default=10, 
                       help='number of top combinations to test exhaustively')
    
    args = parser.parse_args()
    
    # Setup
    eigen_config = {
        'u_n_eigen': args.u,
        'i_n_eigen': args.i,
        'b_n_eigen': args.b
    }
    
    print(f"Interaction-Aware Search on {args.dataset}")
    print(f"Eigenvalues: u={args.u}, i={args.i}, b={args.b}")
    
    # Load dataset
    if args.dataset == 'ml-100k':
        dataset = ML100K()
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
        base_model.user_eigenvals, base_model.user_eigenvecs,
        base_model.item_eigenvals, base_model.item_eigenvecs,
        base_model.bipartite_eigenvals, base_model.bipartite_eigenvecs
    )
    
    # Generate smart combinations
    print("\nGenerating smart combinations based on interaction patterns...")
    combinations = generate_smart_combinations()
    print(f"Testing {len(combinations)} combinations")
    
    results = []
    
    # Phase 1: Test smart combinations
    print("\nPhase 1: Testing interaction-based combinations")
    for i, combo in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing {combo['pattern']}: "
              f"u={combo['user']}, i={combo['item']}, b={combo['bipartite']}")
        
        result = test_combination(search_model, dataset, combo)
        if result:
            results.append(result)
            print(f"  NDCG: {result['ndcg']:.4f}")
    
    # Sort by NDCG
    results.sort(key=lambda x: x['ndcg'], reverse=True)
    
    # Phase 2: Local search around top combinations
    print(f"\n\nPhase 2: Local search around top {args.top_k} combinations")
    
    all_filters = ['original', 'chebyshev', 'jacobi', 'legendre', 'laguerre', 'hermite',
                   'bernstein', 'multiscale', 'bandstop', 'ensemble', 'golden', 'harmonic',
                   'spectral_basis', 'enhanced_basis']
    
    local_search_results = []
    
    for i, top_combo in enumerate(results[:args.top_k]):
        print(f"\n\nLocal search around combination {i+1}:")
        print(f"  Base: u={top_combo['user']}, i={top_combo['item']}, "
              f"b={top_combo['bipartite']} (NDCG: {top_combo['ndcg']:.4f})")
        
        # Try variations by changing one filter at a time
        for view in ['user', 'item', 'bipartite']:
            for new_filter in all_filters:
                # Skip if it's the same as current
                if new_filter == top_combo[view]:
                    continue
                
                # Create variation
                variation = {
                    'user': top_combo['user'],
                    'item': top_combo['item'],
                    'bipartite': top_combo['bipartite'],
                    'pattern': f"local_search_from_{i+1}"
                }
                variation[view] = new_filter
                
                # Skip if we already tested this
                already_tested = any(
                    r['user'] == variation['user'] and 
                    r['item'] == variation['item'] and 
                    r['bipartite'] == variation['bipartite']
                    for r in results + local_search_results
                )
                
                if not already_tested:
                    print(f"  Testing {view}={new_filter}...", end='', flush=True)
                    result = test_combination(search_model, dataset, variation)
                    if result:
                        local_search_results.append(result)
                        print(f" NDCG: {result['ndcg']:.4f}")
                        if result['ndcg'] > top_combo['ndcg']:
                            print(f"    ✓ Improvement! (+{result['ndcg'] - top_combo['ndcg']:.4f})")
    
    # Combine all results
    all_results = results + local_search_results
    all_results.sort(key=lambda x: x['ndcg'], reverse=True)
    
    # Final summary
    print(f"\n\n{'='*60}")
    print("INTERACTION-AWARE SEARCH COMPLETED")
    print(f"\nTop 5 combinations:")
    for i, result in enumerate(all_results[:5]):
        print(f"\n{i+1}. NDCG: {result['ndcg']:.4f}")
        print(f"   User: {result['user']}")
        print(f"   Item: {result['item']}")
        print(f"   Bipartite: {result['bipartite']}")
        print(f"   Pattern: {result['pattern']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"interaction_search_{args.dataset}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'eigenvalues': eigen_config,
            'top_k_local_search': args.top_k,
            'results': all_results,
            'top_combination': all_results[0] if all_results else None,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n\nResults saved to {results_file}")
    
    # Analysis: Find patterns in top combinations
    print("\n\nAnalysis: Common patterns in top 10 combinations")
    top_10 = all_results[:10]
    
    # Count filter frequencies
    for view in ['user', 'item', 'bipartite']:
        filter_counts = {}
        for result in top_10:
            filter_name = result[view]
            filter_counts[filter_name] = filter_counts.get(filter_name, 0) + 1
        
        print(f"\n{view.capitalize()} view - most common filters:")
        for filter_name, count in sorted(filter_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {filter_name}: {count}/10")


if __name__ == "__main__":
    main()