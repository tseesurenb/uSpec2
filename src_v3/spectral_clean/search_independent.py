#!/usr/bin/env python3
"""
Independent filter search - find best filter for each view separately
Much more efficient than testing all combinations!
"""

import numpy as np
import time
import json
import csv
from datetime import datetime
import torch
import world
import utils
from dataloader import Loader, ML100K
from model import SpectralCF
import Procedure
import argparse


class EigenPrecomputedSpectralCF(SpectralCF):
    """SpectralCF that can use precomputed eigendecompositions"""
    
    def __init__(self, adj_mat, config=None):
        super().__init__(adj_mat, config)
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
    
    def train(self):
        """Skip eigendecomposition if already done"""
        if self.eigendecomposition_done:
            print("Using precomputed eigendecompositions")
            return
        super().train()


def test_single_view_filter(model, dataset, view, filter_design):
    """Test a single filter on one view, keeping others as baseline"""
    
    # Set baseline filters for all views
    model.user_filter_design = 'original'
    model.item_filter_design = 'original' 
    model.bipartite_filter_design = 'original'
    
    # Set optimal patterns
    model.user_init_filter = 'smooth'
    model.item_init_filter = 'sharp'
    model.bipartite_init_filter = 'smooth'
    
    # Update only the specific view we're testing
    if view == 'user':
        model.user_filter_design = filter_design
    elif view == 'item':
        model.item_filter_design = filter_design
    elif view == 'bipartite':
        model.bipartite_filter_design = filter_design
    
    # Test the model
    try:
        start_time = time.time()
        results = Procedure.Test(dataset, model, 0, None, world.config['multicore'])
        test_time = time.time() - start_time
        
        return {
            'precision': results['precision'][0],
            'recall': results['recall'][0],
            'ndcg': results['ndcg'][0],
            'test_time': test_time
        }
    except Exception as e:
        print(f"Error testing {view} filter {filter_design}: {e}")
        return None


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Independent Filter Search")
    parser.add_argument('--dataset', type=str, default='gowalla', 
                       choices=['ml-100k', 'gowalla', 'yelp2018', 'amazon-book', 'lastfm'],
                       help='dataset to use')
    parser.add_argument('--u', type=int, default=25, help='user eigenvalues')
    parser.add_argument('--i', type=int, default=200, help='item eigenvalues')
    parser.add_argument('--b', type=int, default=220, help='bipartite eigenvalues')
    
    args = parser.parse_args()
    
    print(f"Starting independent filter search for {args.dataset}...")
    
    # Eigenvalue configuration from command line
    eigen_config = {
        'u_n_eigen': args.u,
        'i_n_eigen': args.i, 
        'b_n_eigen': args.b
    }
    
    print(f"Using eigenvalues: u={eigen_config['u_n_eigen']}, i={eigen_config['i_n_eigen']}, b={eigen_config['b_n_eigen']}")
    
    # Load dataset based on argument
    if args.dataset == 'ml-100k':
        dataset = ML100K()
    elif args.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
        dataset = Loader(path=f"../../data/{args.dataset}")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")
    
    # Create model and do eigendecomposition ONCE
    print("\\nPerforming eigendecomposition (this will take a few minutes)...")
    config = {**world.config, **eigen_config}
    base_model = SpectralCF(dataset.UserItemNet, config)
    
    start_eigen = time.time()
    base_model.train()  # Do eigendecomposition
    eigen_time = time.time() - start_eigen
    print(f"Eigendecomposition completed in {eigen_time:.2f}s")
    
    # All available filters
    filter_designs = ['original', 'chebyshev', 'jacobi', 'legendre', 'laguerre', 'hermite', 
                     'bernstein', 'multiscale', 'bandstop', 'ensemble', 'golden', 'harmonic', 
                     'spectral_basis', 'enhanced_basis']
    
    # Create model that reuses eigendecompositions
    search_model = EigenPrecomputedSpectralCF(dataset.UserItemNet, config)
    search_model.set_precomputed_eigen(
        base_model.user_eigenvals, base_model.user_eigenvecs,
        base_model.item_eigenvals, base_model.item_eigenvecs,
        base_model.bipartite_eigenvals, base_model.bipartite_eigenvecs
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"independent_filter_search_{args.dataset}_{timestamp}.json"
    
    # Results storage
    view_results = {
        'user': [],
        'item': [],
        'bipartite': []
    }
    
    # Test each view independently
    for view in ['user', 'item', 'bipartite']:
        print(f"\\n{'='*50}")
        print(f"TESTING {view.upper()} VIEW FILTERS")
        print(f"{'='*50}")
        
        view_best_ndcg = 0
        view_best_filter = None
        
        for i, filter_design in enumerate(filter_designs):
            print(f"\\nTesting {view} filter {i+1}/{len(filter_designs)}: {filter_design}")
            
            metrics = test_single_view_filter(search_model, dataset, view, filter_design)
            
            if metrics is None:
                print("  Failed")
                continue
            
            result = {
                'filter': filter_design,
                'view': view,
                **metrics
            }
            
            view_results[view].append(result)
            
            # Update best for this view
            if metrics['ndcg'] > view_best_ndcg:
                view_best_ndcg = metrics['ndcg']
                view_best_filter = filter_design
                
            print(f"  NDCG={metrics['ndcg']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, Time={metrics['test_time']:.2f}s")
        
        print(f"\\nBest {view} filter: {view_best_filter} (NDCG={view_best_ndcg:.4f})")
    
    # Test the combination of best filters from each view
    print(f"\\n{'='*60}")
    print("TESTING COMBINATION OF BEST FILTERS")
    print(f"{'='*60}")
    
    # Find best filter for each view
    best_user_filter = max(view_results['user'], key=lambda x: x['ndcg'])['filter']
    best_item_filter = max(view_results['item'], key=lambda x: x['ndcg'])['filter'] 
    best_bipartite_filter = max(view_results['bipartite'], key=lambda x: x['ndcg'])['filter']
    
    print(f"Best combination: user={best_user_filter}, item={best_item_filter}, bipartite={best_bipartite_filter}")
    print(f"Using patterns: user=smooth, item=sharp, bipartite=smooth (fixed for all tests)")
    
    # Test the best combination
    search_model.user_filter_design = best_user_filter
    search_model.item_filter_design = best_item_filter
    search_model.bipartite_filter_design = best_bipartite_filter
    
    # Keep optimal patterns
    search_model.user_init_filter = 'smooth'
    search_model.item_init_filter = 'sharp'
    search_model.bipartite_init_filter = 'smooth'
    
    print("\\nTesting best combination...")
    final_results = Procedure.Test(dataset, search_model, 0, None, world.config['multicore'])
    
    print(f"\\nFINAL RESULTS:")
    print(f"Precision@20: {final_results['precision'][0]:.4f}")
    print(f"Recall@20: {final_results['recall'][0]:.4f}")
    print(f"NDCG@20: {final_results['ndcg'][0]:.4f}")
    print(f"\\nUsing patterns: user={search_model.user_init_filter}, item={search_model.item_init_filter}, bipartite={search_model.bipartite_init_filter}")
    
    # Save all results
    all_results = {
        'eigendecomposition_time': eigen_time,
        'eigen_config': eigen_config,
        'view_results': view_results,
        'best_filters': {
            'user': best_user_filter,
            'item': best_item_filter,
            'bipartite': best_bipartite_filter
        },
        'final_results': {
            'precision': final_results['precision'][0],
            'recall': final_results['recall'][0],
            'ndcg': final_results['ndcg'][0]
        },
        'patterns_used': {
            'user': search_model.user_init_filter,
            'item': search_model.item_init_filter,
            'bipartite': search_model.bipartite_init_filter
        },
        'command_line': f"python main.py --dataset {args.dataset} --u {args.u} --i {args.i} --b {args.b} --uf {best_user_filter.replace('original', 'orig').replace('chebyshev', 'cheby').replace('multiscale', 'multi').replace('bandstop', 'band')} --if {best_item_filter.replace('original', 'orig').replace('chebyshev', 'cheby').replace('multiscale', 'multi').replace('bandstop', 'band')} --bf {best_bipartite_filter.replace('original', 'orig').replace('chebyshev', 'cheby').replace('multiscale', 'multi').replace('bandstop', 'band')}"
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\\nResults saved to: {results_file}")
    
    # Show top 3 for each view
    for view in ['user', 'item', 'bipartite']:
        print(f"\\nTop 3 {view} filters:")
        sorted_view = sorted(view_results[view], key=lambda x: x['ndcg'], reverse=True)
        for i, result in enumerate(sorted_view[:3]):
            print(f"  {i+1}. {result['filter']:15s} - NDCG={result['ndcg']:.4f}")


if __name__ == "__main__":
    main()