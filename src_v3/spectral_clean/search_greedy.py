#!/usr/bin/env python3
"""
Greedy sequential filter search - considers interactions between views
Builds up best combination step by step
"""

import numpy as np
import time
import json
from datetime import datetime
import torch
import world
import utils
from dataloader import Loader, ML100K
from model import SpectralCF
import Procedure
import argparse
from search_independent import EigenPrecomputedSpectralCF


def test_filter_combination(model, dataset, user_filter, item_filter, bipartite_filter):
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
    parser = argparse.ArgumentParser(description="Greedy Sequential Filter Search")
    parser.add_argument('--dataset', type=str, default='gowalla',
                       choices=['ml-100k', 'gowalla', 'yelp2018', 'amazon-book'],
                       help='dataset to use')
    parser.add_argument('--u', type=int, default=25, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=200, help='item eigenvalues')
    parser.add_argument('--b', type=int, default=220, help='bipartite eigenvalues')
    parser.add_argument('--order', type=str, default='uib', 
                       help='search order: uib, iub, bui, etc.')
    parser.add_argument('--in_mat', type=str, default='uib',
                       help='input matrix: u, i, ui, b, ub, or uib')
    
    args = parser.parse_args()
    
    # Setup
    eigen_config = {
        'u_n_eigen': args.u,
        'i_n_eigen': args.i,
        'b_n_eigen': args.b,
        'in_mat': args.in_mat
    }
    
    print(f"Using eigenvalues: u={eigen_config['u_n_eigen']}, i={eigen_config['i_n_eigen']}, b={eigen_config['b_n_eigen']}")
    print(f"Search order: {args.order}")
    print(f"Active views: {args.in_mat}")
    
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
    
    # Set precomputed eigendecompositions based on what's available
    search_model.set_precomputed_eigen(
        user_eigenvals=getattr(base_model, 'user_eigenvals', None),
        user_eigenvecs=getattr(base_model, 'user_eigenvecs', None),
        item_eigenvals=getattr(base_model, 'item_eigenvals', None),
        item_eigenvecs=getattr(base_model, 'item_eigenvecs', None),
        bipartite_eigenvals=getattr(base_model, 'bipartite_eigenvals', None),
        bipartite_eigenvecs=getattr(base_model, 'bipartite_eigenvecs', None)
    )
    
    # Available filters
    filter_designs = ['original', 'chebyshev', 'jacobi', 'legendre', 'laguerre', 'hermite',
                     'bernstein', 'multiscale', 'bandstop', 'ensemble', 'golden', 'harmonic',
                     'spectral_basis', 'enhanced_basis']
    
    # Initialize best combination
    best_combo = {
        'user_filter': 'original',
        'item_filter': 'original', 
        'bipartite_filter': 'original',
        'ndcg': 0.0
    }
    
    # Map order string to view names (only include views that are active in in_mat)
    view_map = {'u': 'user', 'i': 'item', 'b': 'bipartite'}
    search_views = [view_map[c] for c in args.order if c in args.in_mat]
    
    if not search_views:
        print(f"Error: No valid views to search. Order '{args.order}' has no overlap with in_mat '{args.in_mat}'")
        return
    
    results_log = []
    
    # Greedy search
    for step, view in enumerate(search_views):
        print(f"\n{'='*60}")
        print(f"Step {step+1}: Optimizing {view} filter")
        print(f"Current best: user={best_combo['user_filter']}, "
              f"item={best_combo['item_filter']}, bipartite={best_combo['bipartite_filter']}")
        print(f"Current NDCG: {best_combo['ndcg']:.4f}")
        
        view_results = []
        
        # Test each filter for this view
        for filter_design in filter_designs:
            # Create test combination
            test_user = best_combo['user_filter']
            test_item = best_combo['item_filter']
            test_bipartite = best_combo['bipartite_filter']
            
            if view == 'user':
                test_user = filter_design
            elif view == 'item':
                test_item = filter_design
            elif view == 'bipartite':
                test_bipartite = filter_design
            
            print(f"\nTesting {view}={filter_design}...", end='', flush=True)
            result = test_filter_combination(search_model, dataset, 
                                           test_user, test_item, test_bipartite)
            
            if result:
                print(f" NDCG: {result['ndcg']:.4f}")
                view_results.append(result)
                
                # Update best if improved
                if result['ndcg'] > best_combo['ndcg']:
                    best_combo = {
                        'user_filter': test_user,
                        'item_filter': test_item,
                        'bipartite_filter': test_bipartite,
                        'ndcg': result['ndcg'],
                        'recall': result['recall'],
                        'precision': result['precision']
                    }
        
        # Log results for this step
        results_log.append({
            'step': step + 1,
            'view': view,
            'results': view_results,
            'best_so_far': best_combo.copy()
        })
    
    # Final summary
    print(f"\n{'='*60}")
    print("GREEDY SEARCH COMPLETED")
    print(f"Best combination found:")
    print(f"  User filter: {best_combo['user_filter']}")
    print(f"  Item filter: {best_combo['item_filter']}")
    print(f"  Bipartite filter: {best_combo['bipartite_filter']}")
    print(f"  NDCG: {best_combo['ndcg']:.4f}")
    print(f"  Recall: {best_combo['recall']:.4f}")
    print(f"  Precision: {best_combo['precision']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"greedy_search_{args.dataset}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'eigenvalues': eigen_config,
            'search_order': args.order,
            'best_combination': best_combo,
            'search_log': results_log,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()