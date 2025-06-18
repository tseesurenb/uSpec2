#!/usr/bin/env python3
"""
Hyperparameter search for clean spectral model
"""

import numpy as np
import time
import json
import csv
from datetime import datetime
import itertools
from dataloader import ML100K
from model import SpectralCF
import utils


def test_spectral_config(dataset, config):
    """Test a single spectral configuration"""
    try:
        # Create model with config
        model = SpectralCF(dataset.UserItemNet)
        model.u_n_eigen = config['u_n_eigen']
        model.i_n_eigen = config['i_n_eigen'] 
        model.b_n_eigen = config['b_n_eigen']
        
        # Train model
        model.train()
        
        # Simple evaluation on test set
        test_users = list(dataset.testDict.keys())[:50]  # Sample for speed
        all_scores = []
        
        for user in test_users:
            if user in dataset.testDict:
                # Get user rating
                rating = model.getUsersRating([user], 'ml-100k')[0]
                
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
    print("Starting spectral model hyperparameter search...")
    
    # Load dataset
    dataset = ML100K()
    
    # Define search space
    search_space = {
        'u_n_eigen': [10, 25, 50, 75],
        'i_n_eigen': [50, 100, 200, 300], 
        'b_n_eigen': [100, 150, 220, 300]
    }
    
    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    configurations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    print(f"Testing {len(configurations)} configurations...")
    
    # Results storage
    results = []
    best_ndcg = 0
    best_config = None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"spectral_search_ml-100k_{timestamp}.csv"
    json_file = f"spectral_search_ml-100k_{timestamp}.json"
    
    # CSV header
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['u_n_eigen', 'i_n_eigen', 'b_n_eigen', 'precision', 'recall', 'ndcg', 'time'])
    
    # Test each configuration
    for i, config in enumerate(configurations):
        print(f"\nTesting config {i+1}/{len(configurations)}: {config}")
        
        start_time = time.time()
        metrics = test_spectral_config(dataset, config)
        elapsed_time = time.time() - start_time
        
        if metrics is None:
            print("Failed to evaluate this configuration")
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
            
        print(f"Results: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, NDCG={metrics['ndcg']:.4f}, Time={elapsed_time:.2f}s")
        
        # Save to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                config['u_n_eigen'], config['i_n_eigen'], config['b_n_eigen'],
                metrics['precision'], metrics['recall'], metrics['ndcg'], elapsed_time
            ])
    
    # Save all results to JSON
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== SEARCH COMPLETE ===")
    print(f"Best NDCG: {best_ndcg:.4f}")
    print(f"Best config: {best_config}")
    print(f"Results saved to: {csv_file}, {json_file}")
    
    # Show top 5 configurations
    sorted_results = sorted(results, key=lambda x: x['ndcg'], reverse=True)
    print(f"\nTop 5 configurations:")
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. NDCG={result['ndcg']:.4f} - u={result['u_n_eigen']}, i={result['i_n_eigen']}, b={result['b_n_eigen']}")


if __name__ == "__main__":
    main()