#!/usr/bin/env python3
"""
Eigenvalue count search - find optimal eigenvalue counts for each view
Computes maximum eigenvalues once, then tests different subsets
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


def test_eigen_combination(model, dataset, u_count, i_count, b_count, filters):
    """Test a specific eigenvalue count combination with fixed filters"""
    
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
        print(f"Error testing eigen combination u={u_count}, i={i_count}, b={b_count}: {e}")
        return None


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Eigenvalue Count Search")
    parser.add_argument('--dataset', type=str, default='gowalla', 
                       choices=['ml-100k', 'gowalla', 'yelp2018', 'amazon-book', 'lastfm'],
                       help='dataset to use')
    parser.add_argument('--uf', type=str, default='jacobi', help='user filter')
    parser.add_argument('--if_', type=str, default='spectral_basis', help='item filter')
    parser.add_argument('--bf', type=str, default='jacobi', help='bipartite filter')
    
    args = parser.parse_args()
    
    print(f"Starting eigenvalue count search for {args.dataset}...")
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
    
    print(f"Computing MAXIMUM eigendecompositions: u={max_eigen_config['u_n_eigen']}, i={max_eigen_config['i_n_eigen']}, b={max_eigen_config['b_n_eigen']}")
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
    u_range = range(30, 201, 25)  # 30 to 200 step 25
    i_range = range(50, 501, 25)  # 50 to 500 step 25  
    b_range = range(100, 801, 25) # 100 to 800 step 25
    
    total_combinations = len(u_range) * len(i_range) * len(b_range)
    print(f"Testing {total_combinations:,} eigenvalue combinations...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"eigen_search_{args.dataset}_{timestamp}.json"
    csv_file = f"eigen_search_{args.dataset}_{timestamp}.csv"
    
    # Results storage
    results = []
    best_ndcg = 0
    best_config = None
    
    # CSV header
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['u_eigen', 'i_eigen', 'b_eigen', 'precision', 'recall', 'ndcg', 'test_time'])
    
    # Test each eigenvalue combination
    combination_count = 0
    for u_count in u_range:
        for i_count in i_range:
            for b_count in b_range:
                combination_count += 1
                print(f"\nTesting {combination_count}/{total_combinations}: u={u_count}, i={i_count}, b={b_count}")
                
                metrics = test_eigen_combination(search_model, dataset, u_count, i_count, b_count, fixed_filters)
                
                if metrics is None:
                    print("  Failed")
                    continue
                
                result = {
                    'u_eigen': u_count,
                    'i_eigen': i_count,
                    'b_eigen': b_count,
                    'filters': fixed_filters,
                    **metrics
                }
                
                results.append(result)
                
                # Update best
                if metrics['ndcg'] > best_ndcg:
                    best_ndcg = metrics['ndcg']
                    best_config = (u_count, i_count, b_count)
                    
                print(f"  NDCG={metrics['ndcg']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, Time={metrics['test_time']:.2f}s")
                
                # Save to CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        u_count, i_count, b_count,
                        metrics['precision'], metrics['recall'], metrics['ndcg'], metrics['test_time']
                    ])
    
    # Save all results
    all_results = {
        'eigendecomposition_time': eigen_time,
        'max_eigen_config': max_eigen_config,
        'fixed_filters': fixed_filters,
        'search_ranges': {
            'u_range': list(u_range),
            'i_range': list(i_range), 
            'b_range': list(b_range)
        },
        'results': results,
        'best_config': {
            'u_eigen': best_config[0] if best_config else None,
            'i_eigen': best_config[1] if best_config else None,
            'b_eigen': best_config[2] if best_config else None,
            'ndcg': best_ndcg
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EIGENVALUE SEARCH COMPLETE")
    print(f"Total eigendecomposition time: {eigen_time:.2f}s")
    print(f"Best NDCG: {best_ndcg:.4f}")
    if best_config:
        print(f"Best eigenvalue counts: u={best_config[0]}, i={best_config[1]}, b={best_config[2]}")
        print(f"Command to reproduce: python main.py --dataset {args.dataset} --u {best_config[0]} --i {best_config[1]} --b {best_config[2]} --uf {args.uf} --if {args.if_} --bf {args.bf}")
    print(f"Results saved to: {results_file}, {csv_file}")
    
    # Show top 10 combinations
    sorted_results = sorted(results, key=lambda x: x['ndcg'], reverse=True)
    print(f"\nTop 10 eigenvalue combinations:")
    for i, result in enumerate(sorted_results[:10]):
        print(f"{i+1:2d}. NDCG={result['ndcg']:.4f} - u={result['u_eigen']}, i={result['i_eigen']}, b={result['b_eigen']}")


if __name__ == "__main__":
    main()