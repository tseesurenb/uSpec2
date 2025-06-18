#!/usr/bin/env python3
"""
Hierarchical eigenvalue count search - find optimal eigenvalue counts sequentially
Much more efficient than full combinatorial search!
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


class EigenSubsetSpectralCF(SpectralCF):
    """SpectralCF that can use subsets of precomputed eigendecompositions"""
    
    def __init__(self, adj_mat, config=None):
        super().__init__(adj_mat, config)
        self.eigendecomposition_done = False
        self.max_eigendecomposition_done = False
    
    def set_max_precomputed_eigen(self, max_user_eigenvals, max_user_eigenvecs, 
                                 max_item_eigenvals, max_item_eigenvecs,
                                 max_bipartite_eigenvals, max_bipartite_eigenvecs):
        """Set precomputed eigendecompositions with MAXIMUM eigenvalues"""
        self.max_user_eigenvals = max_user_eigenvals
        self.max_user_eigenvecs = max_user_eigenvecs
        self.max_item_eigenvals = max_item_eigenvals
        self.max_item_eigenvecs = max_item_eigenvecs
        self.max_bipartite_eigenvals = max_bipartite_eigenvals
        self.max_bipartite_eigenvecs = max_bipartite_eigenvecs
        self.max_eigendecomposition_done = True
    
    def set_eigen_counts(self, u_count, i_count, b_count):
        """Set how many eigenvalues to use from the precomputed maximum"""
        if not self.max_eigendecomposition_done:
            raise ValueError("Must call set_max_precomputed_eigen first")
            
        # Use subset of precomputed eigenvalues/eigenvectors
        self.user_eigenvals = self.max_user_eigenvals[:u_count]
        self.user_eigenvecs = self.max_user_eigenvecs[:, :u_count]
        
        self.item_eigenvals = self.max_item_eigenvals[:i_count]
        self.item_eigenvecs = self.max_item_eigenvecs[:, :i_count]
        
        self.bipartite_eigenvals = self.max_bipartite_eigenvals[:b_count]
        self.bipartite_eigenvecs = self.max_bipartite_eigenvecs[:, :b_count]
        
        # Update counts
        self.u_n_eigen = u_count
        self.i_n_eigen = i_count
        self.b_n_eigen = b_count
        
        self.eigendecomposition_done = True
    
    def train(self):
        """Skip eigendecomposition if already done"""
        if self.eigendecomposition_done:
            return
        super().train()


def test_eigen_config(model, dataset, u_count, i_count, b_count, filters):
    """Test a specific eigenvalue count configuration with fixed filters"""
    
    # Set eigenvalue counts
    model.set_eigen_counts(u_count, i_count, b_count)
    
    # Set fixed filters
    model.user_filter_design = filters['user']
    model.item_filter_design = filters['item'] 
    model.bipartite_filter_design = filters['bipartite']
    
    # Set optimal patterns
    model.user_init_filter = 'smooth'
    model.item_init_filter = 'sharp'
    model.bipartite_init_filter = 'smooth'
    
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
        print(f"Error testing eigen config u={u_count}, i={i_count}, b={b_count}: {e}")
        return None


def search_best_eigen_for_view(model, dataset, view, search_range, fixed_u, fixed_i, fixed_b, filters):
    """Search for best eigenvalue count for a specific view"""
    best_ndcg = 0
    best_count = None
    results = []
    
    print(f"\nSearching best {view} eigenvalue count...")
    
    for count in search_range:
        if view == 'user':
            u, i, b = count, fixed_i, fixed_b
        elif view == 'item':
            u, i, b = fixed_u, count, fixed_b
        else:  # bipartite
            u, i, b = fixed_u, fixed_i, count
            
        print(f"  Testing {view}={count} (u={u}, i={i}, b={b})")
        
        metrics = test_eigen_config(model, dataset, u, i, b, filters)
        
        if metrics is None:
            print("    Failed")
            continue
            
        result = {
            'view': view,
            'count': count,
            'u_eigen': u,
            'i_eigen': i,
            'b_eigen': b,
            **metrics
        }
        results.append(result)
        
        if metrics['ndcg'] > best_ndcg:
            best_ndcg = metrics['ndcg']
            best_count = count
            
        print(f"    NDCG={metrics['ndcg']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    print(f"  Best {view} eigenvalue count: {best_count} (NDCG={best_ndcg:.4f})")
    
    return best_count, best_ndcg, results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hierarchical Eigenvalue Count Search")
    parser.add_argument('--dataset', type=str, default='gowalla', 
                       choices=['ml-100k', 'gowalla', 'yelp2018', 'amazon-book', 'lastfm'],
                       help='dataset to use')
    parser.add_argument('--uf', type=str, default='jacobi', help='user filter')
    parser.add_argument('--if_', type=str, default='spectral_basis', help='item filter')
    parser.add_argument('--bf', type=str, default='jacobi', help='bipartite filter')
    
    args = parser.parse_args()
    
    print(f"Starting hierarchical eigenvalue count search for {args.dataset}...")
    print(f"Using fixed filters: user={args.uf}, item={args.if_}, bipartite={args.bf}")
    
    # Load dataset based on argument
    if args.dataset == 'ml-100k':
        dataset = ML100K()
    elif args.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
        dataset = Loader(path=f"../../data/{args.dataset}")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")
    
    # Map short filter names to full names
    filter_mapping = {
        'orig': 'original', 'cheby': 'chebyshev', 'jacobi': 'jacobi', 
        'legendre': 'legendre', 'laguerre': 'laguerre', 'hermite': 'hermite',
        'bernstein': 'bernstein', 'multi': 'multiscale', 'band': 'bandstop', 
        'ensemble': 'ensemble', 'golden': 'golden', 'harmonic': 'harmonic',
        'spectral_basis': 'spectral_basis', 'enhanced_basis': 'enhanced_basis'
    }
    
    fixed_filters = {
        'user': filter_mapping.get(args.uf, args.uf),
        'item': filter_mapping.get(args.if_, args.if_),
        'bipartite': filter_mapping.get(args.bf, args.bf)
    }
    
    # Maximum eigenvalue counts (we'll compute these once, then use subsets)
    max_eigen_config = {
        'u_n_eigen': 200,  # Maximum user eigenvalues
        'i_n_eigen': 500,  # Maximum item eigenvalues
        'b_n_eigen': 800   # Maximum bipartite eigenvalues
    }
    
    print(f"\nComputing MAXIMUM eigendecompositions: u={max_eigen_config['u_n_eigen']}, i={max_eigen_config['i_n_eigen']}, b={max_eigen_config['b_n_eigen']}")
    print("This will take several minutes...")
    
    # Create model and do MAXIMUM eigendecomposition ONCE
    config = {**world.config, **max_eigen_config}
    base_model = SpectralCF(dataset.UserItemNet, config)
    
    start_eigen = time.time()
    base_model.train()  # Do maximum eigendecomposition
    eigen_time = time.time() - start_eigen
    print(f"Maximum eigendecomposition completed in {eigen_time:.2f}s")
    
    # Create search model that uses subsets
    search_model = EigenSubsetSpectralCF(dataset.UserItemNet, config)
    search_model.set_max_precomputed_eigen(
        base_model.user_eigenvals, base_model.user_eigenvecs,
        base_model.item_eigenvals, base_model.item_eigenvecs,
        base_model.bipartite_eigenvals, base_model.bipartite_eigenvecs
    )
    
    # Search ranges
    u_range = list(range(30, 201, 25))  # 30 to 200 step 25
    i_range = list(range(50, 501, 25))  # 50 to 500 step 25  
    b_range = list(range(100, 801, 25)) # 100 to 800 step 25
    
    # Starting defaults (middle of ranges)
    default_u = 100
    default_i = 250
    default_b = 400
    
    total_tests = len(u_range) + len(i_range) + len(b_range)
    print(f"\nWill test approximately {total_tests} configurations hierarchically")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"eigen_hierarchical_{args.dataset}_{timestamp}.json"
    
    all_results = {
        'eigendecomposition_time': eigen_time,
        'max_eigen_config': max_eigen_config,
        'fixed_filters': fixed_filters,
        'search_ranges': {
            'u_range': u_range,
            'i_range': i_range, 
            'b_range': b_range
        },
        'hierarchical_results': {}
    }
    
    print(f"\n{'='*60}")
    print("PHASE 1: Find best USER eigenvalue count")
    print(f"{'='*60}")
    best_u, best_u_ndcg, u_results = search_best_eigen_for_view(
        search_model, dataset, 'user', u_range, 
        default_u, default_i, default_b, fixed_filters
    )
    all_results['hierarchical_results']['user_search'] = {
        'results': u_results,
        'best_count': best_u,
        'best_ndcg': best_u_ndcg
    }
    
    print(f"\n{'='*60}")
    print("PHASE 2: Find best ITEM eigenvalue count (with fixed best user)")
    print(f"{'='*60}")
    best_i, best_i_ndcg, i_results = search_best_eigen_for_view(
        search_model, dataset, 'item', i_range,
        best_u, default_i, default_b, fixed_filters
    )
    all_results['hierarchical_results']['item_search'] = {
        'results': i_results,
        'best_count': best_i,
        'best_ndcg': best_i_ndcg,
        'fixed_u': best_u
    }
    
    print(f"\n{'='*60}")
    print("PHASE 3: Find best BIPARTITE eigenvalue count (with fixed best user & item)")
    print(f"{'='*60}")
    best_b, best_b_ndcg, b_results = search_best_eigen_for_view(
        search_model, dataset, 'bipartite', b_range,
        best_u, best_i, default_b, fixed_filters
    )
    all_results['hierarchical_results']['bipartite_search'] = {
        'results': b_results,
        'best_count': best_b,
        'best_ndcg': best_b_ndcg,
        'fixed_u': best_u,
        'fixed_i': best_i
    }
    
    # Final test with all best eigenvalue counts
    print(f"\n{'='*60}")
    print("FINAL TEST: Best hierarchical configuration")
    print(f"{'='*60}")
    print(f"Testing best configuration: u={best_u}, i={best_i}, b={best_b}")
    
    final_metrics = test_eigen_config(search_model, dataset, best_u, best_i, best_b, fixed_filters)
    
    all_results['final_result'] = {
        'u_eigen': best_u,
        'i_eigen': best_i,
        'b_eigen': best_b,
        'metrics': final_metrics
    }
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"HIERARCHICAL SEARCH COMPLETE")
    print(f"Total eigendecomposition time: {eigen_time:.2f}s")
    print(f"Total tests performed: {len(u_results) + len(i_results) + len(b_results) + 1}")
    print(f"Best configuration: u={best_u}, i={best_i}, b={best_b}")
    if final_metrics:
        print(f"Final NDCG: {final_metrics['ndcg']:.4f}")
        print(f"Command to reproduce: python main.py --dataset {args.dataset} --u {best_u} --i {best_i} --b {best_b} --uf {args.uf} --if {args.if_} --bf {args.bf}")
    print(f"Results saved to: {results_file}")
    
    # Show progression through phases
    print(f"\nProgression through phases:")
    print(f"1. User search: Best u={best_u} with NDCG={best_u_ndcg:.4f}")
    print(f"2. Item search: Best i={best_i} with NDCG={best_i_ndcg:.4f}")  
    print(f"3. Bipartite search: Best b={best_b} with NDCG={best_b_ndcg:.4f}")
    if final_metrics:
        print(f"4. Final test: NDCG={final_metrics['ndcg']:.4f}")


if __name__ == "__main__":
    main()