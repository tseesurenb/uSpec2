#!/usr/bin/env python3
"""
Quick test of promising configurations for sparse data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SpectralCF
from dataloader import Loader
import Procedure
import world


def test_sparse_configuration(u_eigen, i_eigen, b_eigen, user_filter, item_filter, bipartite_filter, 
                             user_init, item_init, bipartite_init, dataset_name="gowalla"):
    """Test a specific configuration"""
    
    print(f"\\n=== Testing Configuration ===")
    print(f"Eigenvals: u={u_eigen}, i={i_eigen}, b={b_eigen}")
    print(f"Filters: user={user_filter}, item={item_filter}, bipartite={bipartite_filter}")
    print(f"Inits: user={user_init}, item={item_init}, bipartite={bipartite_init}")
    
    # Load dataset
    if dataset_name == "gowalla_subset":
        dataset = Loader(path="../../data/gowalla_subset")
    else:
        dataset = Loader(path="../../data/gowalla")
    
    # Create model 
    model = SpectralCF(dataset.UserItemNet)
    
    # Update configuration BEFORE any operations
    model.u_n_eigen = u_eigen
    model.i_n_eigen = i_eigen
    model.b_n_eigen = b_eigen
    
    model.user_filter_design = user_filter
    model.item_filter_design = item_filter
    model.bipartite_filter_design = bipartite_filter
    
    model.user_init_filter = user_init
    model.item_init_filter = item_init
    model.bipartite_init_filter = bipartite_init
    
    # Print actual config being used
    print(f"Actually using: u={model.u_n_eigen}, i={model.i_n_eigen}, b={model.b_n_eigen}")
    
    # Train and test
    model.train()
    
    # Test
    epoch = 0
    results = Procedure.Test(dataset, model, epoch, None, world.config['multicore'])
    
    print(f"Results: Precision={results['precision'][0]:.4f}, Recall={results['recall'][0]:.4f}, NDCG={results['ndcg'][0]:.4f}")
    
    return results['ndcg'][0]


def main():
    print("Testing promising sparse data configurations...")
    
    # First create subset if it doesn't exist
    subset_path = "../../data/gowalla_subset"
    if not os.path.exists(subset_path):
        print("Creating Gowalla subset...")
        import subprocess
        result = subprocess.run(['python', 'create_gowalla_subset.py'], capture_output=True, text=True)
        print(result.stdout)
    
    # Test configurations that typically work well for sparse data
    configs = [
        # Config 1: Low eigenvals, simple filters
        {
            'u_eigen': 10, 'i_eigen': 50, 'b_eigen': 75,
            'user_filter': 'original', 'item_filter': 'original', 'bipartite_filter': 'original',
            'user_init': 'smooth', 'item_init': 'sharp', 'bipartite_init': 'smooth'
        },
        
        # Config 2: Medium eigenvals, proven filters  
        {
            'u_eigen': 15, 'i_eigen': 100, 'b_eigen': 150,
            'user_filter': 'bandstop', 'item_filter': 'chebyshev', 'bipartite_filter': 'multiscale',
            'user_init': 'bandpass', 'item_init': 'sharp', 'bipartite_init': 'smooth'
        },
        
        # Config 3: Focus on bipartite (important for sparse)
        {
            'u_eigen': 15, 'i_eigen': 75, 'b_eigen': 200,
            'user_filter': 'original', 'item_filter': 'original', 'bipartite_filter': 'multiscale',
            'user_init': 'smooth', 'item_init': 'bandpass', 'bipartite_init': 'band_stop'
        },
        
        # Config 4: Minimal eigenvals for very sparse data
        {
            'u_eigen': 8, 'i_eigen': 40, 'b_eigen': 60,
            'user_filter': 'original', 'item_filter': 'original', 'bipartite_filter': 'original',
            'user_init': 'smooth', 'item_init': 'sharp', 'bipartite_init': 'smooth'
        }
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\\n{'='*50}")
        print(f"TESTING CONFIG {i}/4")
        
        try:
            ndcg = test_sparse_configuration(
                config['u_eigen'], config['i_eigen'], config['b_eigen'],
                config['user_filter'], config['item_filter'], config['bipartite_filter'],
                config['user_init'], config['item_init'], config['bipartite_init'],
                "gowalla_subset"
            )
            results.append((config, ndcg))
        except Exception as e:
            print(f"Failed with error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((config, 0.0))
    
    # Show best results
    print(f"\\n{'='*50}")
    print("SUMMARY OF RESULTS:")
    results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (config, ndcg) in enumerate(results, 1):
        print(f"\\n{i}. NDCG = {ndcg:.4f}")
        print(f"   Eigenvals: u={config['u_eigen']}, i={config['i_eigen']}, b={config['b_eigen']}")
        print(f"   Filters: {config['user_filter']}, {config['item_filter']}, {config['bipartite_filter']}")
    
    # Suggest best config for full Gowalla
    if results:
        best_config = results[0][0]
        print(f"\\n{'='*50}")
        print("RECOMMENDED CONFIG FOR FULL GOWALLA:")
        print(f"u_n_eigen = {best_config['u_eigen']}")
        print(f"i_n_eigen = {best_config['i_eigen']}")
        print(f"b_n_eigen = {best_config['b_eigen']}")
        print(f"user_filter_design = '{best_config['user_filter']}'")
        print(f"item_filter_design = '{best_config['item_filter']}'") 
        print(f"bipartite_filter_design = '{best_config['bipartite_filter']}'")
        print(f"user_init_filter = '{best_config['user_init']}'")
        print(f"item_init_filter = '{best_config['item_init']}'")
        print(f"bipartite_init_filter = '{best_config['bipartite_init']}'")


if __name__ == "__main__":
    main()