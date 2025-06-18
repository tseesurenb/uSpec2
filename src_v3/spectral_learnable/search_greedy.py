#!/usr/bin/env python3
"""
Greedy sequential filter search for learnable model
Considers interactions between views while optimizing
"""

import numpy as np
import time
import json
from datetime import datetime
import torch
import world
import utils
from dataloader import Loader
from model_simplified import SimplifiedSpectralCF
import procedure
import argparse


class PrecomputedSpectralCF(SimplifiedSpectralCF):
    """SpectralCF that can use precomputed eigendecompositions"""
    
    def __init__(self, adj_mat, config=None):
        # Skip the automatic setup_spectral_filters call
        nn.Module.__init__(self)
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.filter = self.config.get('filter', 'uib')
        self.dataset = self.config.get('dataset', 'unknown')
        
        # Convert adjacency matrix
        adj_dense = adj_mat.toarray() if sp.issparse(adj_mat) else adj_mat
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32).to(self.device))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Store original adjacency matrix
        if sp.issparse(adj_mat):
            self.adj_mat = adj_mat.tocsr()
        else:
            self.adj_mat = sp.csr_matrix(adj_mat)
        
        # Setup cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), '../cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Eigenvalue counts
        self.u_n_eigen = self.config.get('u_n_eigen', 8)
        self.i_n_eigen = self.config.get('i_n_eigen', 40)
        self.b_n_eigen = self.config.get('b_n_eigen', 60)
        
        # Filter mappings
        self.filter_mapping = {
            'orig': 'original', 'cheby': 'chebyshev', 'jacobi': 'jacobi', 
            'legendre': 'legendre', 'laguerre': 'laguerre', 'hermite': 'hermite',
            'bernstein': 'bernstein', 'multi': 'multiscale', 'band': 'bandstop', 
            'ensemble': 'ensemble', 'golden': 'golden', 'harmonic': 'harmonic',
            'spectral_basis': 'spectral_basis', 'enhanced_basis': 'enhanced_basis'
        }
        
        self.init_mapping = {
            'smooth': 'smooth', 'sharp': 'sharp', 'bandpass': 'bandpass',
            'golden': 'golden_036', 'butter': 'butterworth', 'gauss': 'gaussian',
            'stop': 'band_stop', 'notch': 'notch'
        }
        
        self.eigendecomposition_done = False
    
    def set_precomputed_eigen(self, user_eigenvals=None, user_eigenvecs=None, 
                             item_eigenvals=None, item_eigenvecs=None,
                             bipartite_eigenvals=None, bipartite_eigenvecs=None):
        """Set precomputed eigendecompositions"""
        if user_eigenvals is not None:
            self.user_eigenvals = user_eigenvals
            self.user_eigenvecs = user_eigenvecs
        if item_eigenvals is not None:
            self.item_eigenvals = item_eigenvals
            self.item_eigenvecs = item_eigenvecs
        if bipartite_eigenvals is not None:
            self.bipartite_eigenvals = bipartite_eigenvals
            self.bipartite_eigenvecs = bipartite_eigenvecs
        self.eigendecomposition_done = True


def test_filter_combination(model, dataset, user_filter, item_filter, bipartite_filter):
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
    parser = argparse.ArgumentParser(description="Greedy Sequential Filter Search - Learnable Model")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'gowalla', 'yelp2018', 'amazon-book'],
                       help='dataset to use')
    parser.add_argument('--u', type=int, default=8, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=40, help='item eigenvalues')
    parser.add_argument('--b', type=int, default=60, help='bipartite eigenvalues')
    parser.add_argument('--order', type=str, default='uib', 
                       help='search order: uib, iub, bui, etc.')
    parser.add_argument('--in_mat', type=str, default='uib',
                       help='which views to include in search')
    
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
    
    print(f"Using eigenvalues: u={args.u}, i={args.i}, b={args.b}")
    print(f"Search order: {args.order}")
    print(f"Active views: {args.in_mat}")
    
    # Load dataset
    dataset = Loader(args.dataset)
    
    # Create base model and do eigendecomposition
    print("\nPerforming eigendecomposition...")
    base_model = SimplifiedSpectralCF(dataset.UserItemNet, world.config)
    
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
    
    # Available filters (short names)
    filter_designs = ['orig', 'cheby', 'jacobi', 'legendre', 'laguerre', 'hermite',
                     'bernstein', 'multi', 'band', 'ensemble', 'golden', 'harmonic',
                     'spectral_basis', 'enhanced_basis']
    
    # Initialize best combination
    best_combo = {
        'user_filter': 'orig',
        'item_filter': 'orig', 
        'bipartite_filter': 'orig',
        'ndcg': 0.0
    }
    
    # Map order string to view names (only include active views)
    view_map = {'u': 'user', 'i': 'item', 'b': 'bipartite'}
    search_views = [view_map[c] for c in args.order if c in args.in_mat]
    
    results_log = []
    
    # Test baseline first
    print("\nTesting baseline (all original filters)...")
    baseline_result = test_filter_combination(search_model, dataset, 'orig', 'orig', 'orig')
    if baseline_result:
        best_combo['ndcg'] = baseline_result['ndcg']
        best_combo['recall'] = baseline_result['recall']
        best_combo['precision'] = baseline_result['precision']
        print(f"Baseline NDCG: {baseline_result['ndcg']:.4f}")
    
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
            # Skip if we already tested this combination
            test_user = best_combo['user_filter']
            test_item = best_combo['item_filter']
            test_bipartite = best_combo['bipartite_filter']
            
            if view == 'user':
                test_user = filter_design
            elif view == 'item':
                test_item = filter_design
            elif view == 'bipartite':
                test_bipartite = filter_design
            
            # Skip if this is the current best
            if (test_user == best_combo['user_filter'] and 
                test_item == best_combo['item_filter'] and 
                test_bipartite == best_combo['bipartite_filter']):
                continue
            
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
    results_file = f"greedy_search_learnable_{args.dataset}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'eigenvalues': {'u': args.u, 'i': args.i, 'b': args.b},
            'search_order': args.order,
            'active_views': args.in_mat,
            'best_combination': best_combo,
            'search_log': results_log,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Print command to run the best configuration
    print(f"\nTo run the best configuration:")
    print(f"python main.py --dataset {args.dataset} --in_mat {args.in_mat} "
          f"--u {args.u} --i {args.i} --b {args.b} "
          f"--uf {best_combo['user_filter']} --if_ {best_combo['item_filter']} "
          f"--bf {best_combo['bipartite_filter']}")


if __name__ == "__main__":
    main()