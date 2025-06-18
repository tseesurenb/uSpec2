#!/usr/bin/env python3
"""
Efficient search that does eigendecomposition once, then tests all filter combinations
"""

import numpy as np
import time
import json
import csv
from datetime import datetime
import itertools
import torch
import world
import utils
from dataloader import Loader
from model import SpectralCF
import Procedure


class EigenPrecomputedSpectralCF(SpectralCF):
    """SpectralCF that can use precomputed eigendecompositions"""
    
    def __init__(self, adj_mat, config=None):
        super().__init__(adj_mat, config)
        self.eigendecomposition_done = False
    
    def set_precomputed_eigen(self, user_eigenvals, user_eigenvecs, 
                             item_eigenvals, item_eigenvecs,
                             bipartite_eigenvals, bipartite_eigenvecs):
        """Set precomputed eigendecompositions"""
        self.user_eigenvals = user_eigenvals
        self.user_eigenvecs = user_eigenvecs
        self.item_eigenvals = item_eigenvals
        self.item_eigenvecs = item_eigenvecs
        self.bipartite_eigenvals = bipartite_eigenvals
        self.bipartite_eigenvecs = bipartite_eigenvecs
        self.eigendecomposition_done = True
    
    def train(self):
        """Skip eigendecomposition if already done"""
        if self.eigendecomposition_done:
            print("Using precomputed eigendecompositions")
            return
        super().train()


def test_filter_combination(model, dataset, filter_config):
    """Test a single filter combination on the model with precomputed eigendecompositions"""
    
    # Update filter settings without recomputing eigendecompositions
    model.user_filter_design = filter_config['user_filter_design']
    model.item_filter_design = filter_config['item_filter_design'] 
    model.bipartite_filter_design = filter_config['bipartite_filter_design']
    
    model.user_init_filter = filter_config['user_init_filter']
    model.item_init_filter = filter_config['item_init_filter']
    model.bipartite_init_filter = filter_config['bipartite_init_filter']
    
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
        print(f"Error testing filter combination: {e}")
        return None


def main():
    print("Starting efficient filter search...")
    
    # Fixed eigenvalue configuration (use your current best)
    eigen_config = {
        'u_n_eigen': 80,
        'i_n_eigen': 400, 
        'b_n_eigen': 600
    }
    
    print(f"Using eigenvalues: u={eigen_config['u_n_eigen']}, i={eigen_config['i_n_eigen']}, b={eigen_config['b_n_eigen']}")
    
    # Load dataset
    dataset = Loader(path="../../data/gowalla")
    
    # Create model and do eigendecomposition ONCE
    print("\\nPerforming eigendecomposition (this will take a few minutes)...")
    config = {**world.config, **eigen_config}
    base_model = SpectralCF(dataset.UserItemNet, config)
    
    start_eigen = time.time()
    base_model.train()  # Do eigendecomposition
    eigen_time = time.time() - start_eigen
    print(f"Eigendecomposition completed in {eigen_time:.2f}s")
    
    # Define COMPLETE filter search space - ALL available filters
    filter_designs = ['orig', 'cheby', 'jacobi', 'legendre', 'laguerre', 'hermite', 'bernstein', 
                     'multi', 'band', 'ensemble', 'golden', 'harmonic', 'spectral_basis', 'enhanced_basis']
    
    # Use OPTIMAL init patterns from ML-100K success (fixed)
    optimal_user_init = 'smooth'
    optimal_item_init = 'sharp' 
    optimal_bipartite_init = 'smooth'
    
    print(f"Using optimal init patterns: user={optimal_user_init}, item={optimal_item_init}, bipartite={optimal_bipartite_init}")
    
    # Mapping dictionary for filters
    filter_mapping = {
        'orig': 'original', 'cheby': 'chebyshev', 'jacobi': 'jacobi', 
        'legendre': 'legendre', 'laguerre': 'laguerre', 'hermite': 'hermite',
        'bernstein': 'bernstein', 'multi': 'multiscale', 'band': 'bandstop', 
        'ensemble': 'ensemble', 'golden': 'golden', 'harmonic': 'harmonic',
        'spectral_basis': 'spectral_basis', 'enhanced_basis': 'enhanced_basis'
    }
    
    # Generate ALL possible filter combinations with FIXED optimal patterns
    filter_combinations = []
    total_combinations = len(filter_designs) ** 3  # Only filter combinations
    print(f"Generating {total_combinations:,} filter combinations (patterns fixed)...")
    
    for uf in filter_designs:  # ALL user filters
        for if_design in filter_designs:  # ALL item filters  
            for bf in filter_designs:  # ALL bipartite filters
                filter_combinations.append({
                    'user_filter_design': filter_mapping[uf],
                    'item_filter_design': filter_mapping[if_design],
                    'bipartite_filter_design': filter_mapping[bf],
                    'user_init_filter': optimal_user_init,
                    'item_init_filter': optimal_item_init,
                    'bipartite_init_filter': optimal_bipartite_init,
                    'short_name': f"{uf}-{if_design}-{bf}"
                })
    
    print(f"Testing {len(filter_combinations)} filter combinations...")
    
    # Create model that reuses eigendecompositions
    search_model = EigenPrecomputedSpectralCF(dataset.UserItemNet, config)
    search_model.set_precomputed_eigen(
        base_model.user_eigenvals, base_model.user_eigenvecs,
        base_model.item_eigenvals, base_model.item_eigenvecs,
        base_model.bipartite_eigenvals, base_model.bipartite_eigenvecs
    )
    
    # Results storage
    results = []
    best_ndcg = 0
    best_config = None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"filter_search_gowalla_{timestamp}.csv"
    json_file = f"filter_search_gowalla_{timestamp}.json"
    
    # CSV header
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['combination', 'user_filter', 'item_filter', 'bipartite_filter',
                        'precision', 'recall', 'ndcg', 'test_time'])
    
    # Test each filter combination
    print("\\nTesting filter combinations...")
    for i, filter_config in enumerate(filter_combinations):
        print(f"\\nTesting {i+1}/{len(filter_combinations)}: {filter_config['short_name']}")
        
        metrics = test_filter_combination(search_model, dataset, filter_config)
        
        if metrics is None:
            print("  Failed")
            continue
            
        result = {
            **filter_config,
            **metrics
        }
        
        results.append(result)
        
        # Update best
        if metrics['ndcg'] > best_ndcg:
            best_ndcg = metrics['ndcg']
            best_config = filter_config
            
        print(f"  NDCG={metrics['ndcg']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, Time={metrics['test_time']:.2f}s")
        
        # Save to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                filter_config['short_name'],
                filter_config['user_filter_design'], filter_config['item_filter_design'], filter_config['bipartite_filter_design'],
                metrics['precision'], metrics['recall'], metrics['ndcg'], metrics['test_time']
            ])
    
    # Save all results to JSON
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n{'='*60}")
    print(f"FILTER SEARCH COMPLETE")
    print(f"Total eigendecomposition time: {eigen_time:.2f}s")
    print(f"Best NDCG: {best_ndcg:.4f}")
    print(f"Best combination: {best_config['short_name'] if best_config else 'None'}")
    if best_config:
        print(f"Best filters: {best_config['user_filter_design']}, {best_config['item_filter_design']}, {best_config['bipartite_filter_design']}")
        print(f"Using patterns: {optimal_user_init}, {optimal_item_init}, {optimal_bipartite_init} (fixed)")
    print(f"Results saved to: {csv_file}, {json_file}")
    
    # Show top 10 combinations
    sorted_results = sorted(results, key=lambda x: x['ndcg'], reverse=True)
    print(f"\\nTop 10 filter combinations:")
    for i, result in enumerate(sorted_results[:10]):
        print(f"{i+1:2d}. NDCG={result['ndcg']:.4f} - {result['short_name']}")


if __name__ == "__main__":
    main()