#!/usr/bin/env python3
"""
Test script for temperature scaling in user similarity
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from config import parse_args, get_config
from dataloader import Loader
from learnable_model import SpectralCFLearnable
import torch
import numpy as np

def test_temperature_scaling():
    """Test temperature scaling functionality"""
    print("=" * 60)
    print("Testing Temperature Scaling for User Similarity")
    print("=" * 60)
    
    # Override args for testing
    import argparse
    args = argparse.Namespace(
        dataset='gowalla',
        full_training=True,
        filter='u',  # Only user view to test temperature
        filter_type='bernstein',
        filter_order=8,
        u=25,
        i=200,
        b=220,
        user_init='smooth',
        item_init='sharp',
        bipartite_init='smooth',
        user_lr=0.1,
        item_lr=0.01,
        bipartite_lr=0.05,
        user_decay=1e-4,
        item_decay=1e-3,
        bipartite_decay=5e-4,
        epochs=1,
        train_batch=1000,
        test_batch=500,
        neg_ratio=4,
        loss='mse',
        optimizer='adam',
        scheduler='none',
        patience=5,
        topks='[20]',
        eval_freq=5,
        seed=2020,
        model='lgn',
        layer=0,
        eval_early_stop=True,
        device='cpu',
        testbatch=100,
        multicore=0,
        pretrain=0,
        verbose=1,
        early_stop_metric='recall',
        beta=0.01,
        adam_weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        learnable_gamma=False,
        use_two_hop=False,
        two_hop_weight=0.3,
        raw_only=False,
        user_temperature=0.1,  # Temperature parameter
        exp_name='temp_test',
        keep_prob=0.6,
        a_fold=100,
        bigdata=False
    )
    
    # Load dataset
    print("Loading dataset...")
    config = get_config(args)
    dataset = Loader(config)
    
    # Test with different temperature values
    temperatures = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    for temp in temperatures:
        print(f"\n--- Testing with temperature = {temp} ---")
        config['user_temperature'] = temp
        
        # Create model
        model = SpectralCFLearnable(dataset.Graph, config)
        
        # Check the similarity computation
        user_sim = model._compute_user_similarity()
        
        # Analyze similarity distribution
        if hasattr(user_sim, 'data'):  # Sparse matrix
            sim_values = user_sim.data
        else:
            sim_values = user_sim[user_sim > 0]
        
        print(f"Non-zero similarities: {len(sim_values)}")
        print(f"Min similarity: {np.min(sim_values):.6f}")
        print(f"Max similarity: {np.max(sim_values):.6f}")
        print(f"Mean similarity: {np.mean(sim_values):.6f}")
        print(f"Std similarity: {np.std(sim_values):.6f}")
        
        # Show histogram of similarity values
        hist, bins = np.histogram(sim_values, bins=10)
        print(f"Similarity distribution:")
        for i in range(len(hist)):
            print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}]: {hist[i]} ({hist[i]/len(sim_values)*100:.1f}%)")

if __name__ == "__main__":
    test_temperature_scaling()