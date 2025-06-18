#!/usr/bin/env python3
"""
Hyperparameter search optimized for sparse datasets like Gowalla
"""

import numpy as np
import time
import json
import csv
from datetime import datetime
import itertools
from dataloader import Loader
from model import SpectralCF
import utils


def test_sparse_config(dataset, config):
    """Test a single spectral configuration on sparse data"""
    try:
        # Create model with config
        model = SpectralCF(dataset.UserItemNet)
        model.u_n_eigen = config['u_n_eigen']
        model.i_n_eigen = config['i_n_eigen'] 
        model.b_n_eigen = config['b_n_eigen']
        
        # Set filter designs
        model.user_filter_design = config['user_filter']
        model.item_filter_design = config['item_filter']
        model.bipartite_filter_design = config['bipartite_filter']
        
        model.user_init_filter = config['user_init']
        model.item_init_filter = config['item_init']
        model.bipartite_init_filter = config['bipartite_init']
        
        # Train model
        model.train()
        
        # Quick evaluation on sample of test users
        test_users = list(dataset.testDict.keys())[:100]  # Sample for speed
        all_scores = []
        
        for user in test_users:
            if user in dataset.testDict and len(dataset.testDict[user]) > 0:
                # Get user rating
                rating = model.getUsersRating([user], 'gowalla_subset')[0]
                
                # Get ground truth
                pos_items = dataset.getUserPosItems([user])[0]
                test_items = dataset.testDict[user]
                
                # Exclude training items
                rating[pos_items] = -np.inf
                
                # Get top-20 recommendations
                top_items = np.argsort(rating)[-20:][::-1]
                
                # Calculate metrics
                hits = len(set(top_items) & set(test_items))
                precision = hits / 20
                recall = hits / len(test_items) if len(test_items) > 0 else 0
                
                # NDCG@20
                dcg = 0
                for i, item in enumerate(top_items):
                    if item in test_items:
                        dcg += 1 / np.log2(i + 2)
                
                # IDCG 
                idcg = sum(1 / np.log2(i + 2) for i in range(min(len(test_items), 20)))
                ndcg = dcg / idcg if idcg > 0 else 0
                
                all_scores.append({'precision': precision, 'recall': recall, 'ndcg': ndcg})
        
        if not all_scores:
            return None
            
        # Average metrics
        avg_precision = np.mean([s['precision'] for s in all_scores])
        avg_recall = np.mean([s['recall'] for s in all_scores])
        avg_ndcg = np.mean([s['ndcg'] for s in all_scores])
        
        return {
            'precision': avg_precision,
            'recall': avg_recall, 
            'ndcg': avg_ndcg
        }
        
    except Exception as e:
        print(f"Error with config {config}: {e}")
        return None


def main():
    print("Creating Gowalla subset first...")
    
    # Create subset
    import subprocess
    result = subprocess.run(['python', 'create_gowalla_subset.py'], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    print("\\nStarting sparse dataset hyperparameter search...")
    
    # Load subset dataset
    dataset = Loader(path="../../data/gowalla_subset")
    
    # Define search space optimized for sparse data
    search_space = {
        # Lower eigenvalue counts for sparse data
        'u_n_eigen': [10, 15, 25],
        'i_n_eigen': [50, 100, 150], 
        'b_n_eigen': [75, 150, 200],
        
        # Filter designs that work well on sparse data
        'user_filter': ['original', 'bandstop', 'multiscale'],
        'item_filter': ['original', 'chebyshev'],
        'bipartite_filter': ['original', 'multiscale'],
        
        # Initialization patterns for sparse data
        'user_init': ['smooth', 'bandpass'],
        'item_init': ['sharp', 'bandpass'],
        'bipartite_init': ['smooth', 'band_stop']
    }
    
    # Generate combinations (limit to reasonable number)
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    # Generate all combinations but sample subset for speed
    all_combinations = list(itertools.product(*values))
    
    # Sample 50 configurations for faster search
    if len(all_combinations) > 50:
        configurations = [dict(zip(keys, combo)) for combo in 
                         np.random.choice(len(all_combinations), 50, replace=False)]
        configurations = [dict(zip(keys, all_combinations[i])) for i in 
                         np.random.choice(len(all_combinations), 50, replace=False)]
    else:
        configurations = [dict(zip(keys, combo)) for combo in all_combinations]
    
    print(f"Testing {len(configurations)} configurations on Gowalla subset...")
    
    # Results storage
    results = []
    best_ndcg = 0
    best_config = None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"sparse_search_gowalla_{timestamp}.csv"
    json_file = f"sparse_search_gowalla_{timestamp}.json"
    
    # CSV header
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['u_n_eigen', 'i_n_eigen', 'b_n_eigen', 
                        'user_filter', 'item_filter', 'bipartite_filter',
                        'user_init', 'item_init', 'bipartite_init',
                        'precision', 'recall', 'ndcg', 'time'])
    
    # Test each configuration
    for i, config in enumerate(configurations):
        print(f"\\nTesting config {i+1}/{len(configurations)}:")
        print(f"  Eigenvals: u={config['u_n_eigen']}, i={config['i_n_eigen']}, b={config['b_n_eigen']}")
        print(f"  Filters: {config['user_filter']}, {config['item_filter']}, {config['bipartite_filter']}")
        print(f"  Inits: {config['user_init']}, {config['item_init']}, {config['bipartite_init']}")
        
        start_time = time.time()
        metrics = test_sparse_config(dataset, config)
        elapsed_time = time.time() - start_time
        
        if metrics is None:
            print("  Failed to evaluate this configuration")
            continue
            
        result = {
            **config,
            **metrics,
            'time': elapsed_time
        }
        
        results.append(result)
        
        # Update best
        if metrics['ndcg'] > best_ndcg:
            best_ndcg = metrics['ndcg']
            best_config = config
            
        print(f"  Results: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, NDCG={metrics['ndcg']:.4f}, Time={elapsed_time:.2f}s")
        
        # Save to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                config['u_n_eigen'], config['i_n_eigen'], config['b_n_eigen'],
                config['user_filter'], config['item_filter'], config['bipartite_filter'],
                config['user_init'], config['item_init'], config['bipartite_init'],
                metrics['precision'], metrics['recall'], metrics['ndcg'], elapsed_time
            ])
    
    # Save all results to JSON
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n=== SPARSE DATA SEARCH COMPLETE ===")
    print(f"Best NDCG: {best_ndcg:.4f}")
    print(f"Best config: {best_config}")
    print(f"Results saved to: {csv_file}, {json_file}")
    
    # Show top 5 configurations
    sorted_results = sorted(results, key=lambda x: x['ndcg'], reverse=True)
    print(f"\\nTop 5 configurations for sparse data:")
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. NDCG={result['ndcg']:.4f}")
        print(f"   Eigenvals: u={result['u_n_eigen']}, i={result['i_n_eigen']}, b={result['b_n_eigen']}")
        print(f"   Filters: {result['user_filter']}, {result['item_filter']}, {result['bipartite_filter']}")


if __name__ == "__main__":
    main()