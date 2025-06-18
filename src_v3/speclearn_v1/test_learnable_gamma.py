#!/usr/bin/env python3
"""
Test script for learnable gamma parameter in spectral CF
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from config import parse_args, get_config
from dataloader import Loader
from learnable_model import SpectralCFLearnable
import torch

def test_learnable_gamma():
    """Test learnable gamma functionality"""
    print("=" * 60)
    print("Testing Learnable Gamma Parameter for Item View")
    print("=" * 60)
    
    # Override args for testing
    import argparse
    args = argparse.Namespace(
        dataset='gowalla',
        full_training=True,
        filter='i',  # Only item view
        filter_type='bernstein',
        filter_order=8,
        u=25,
        i=200,
        b=220,
        user_init='smooth',
        item_init='linear_dec',  # Use optimal pattern from search
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
        device='auto',
        verbose=1,
        exp_name='gamma_test',
        save_model=False,
        log_filters=False,
        use_two_hop=False,
        two_hop_weight=0.3,
        raw_only=False,
        learnable_gamma=True  # Enable learnable gamma
    )
    
    config = get_config(args)
    
    # Load dataset
    print("Loading dataset...")
    dataset = Loader(config)
    
    # Test 1: Model without learnable gamma
    print("\n" + "=" * 40)
    print("Test 1: Standard model (gamma=0.5)")
    print("=" * 40)
    
    config_standard = config.copy()
    config_standard['learnable_gamma'] = False
    
    model_standard = SpectralCFLearnable(dataset.UserItemNet, config_standard)
    
    # Test 2: Model with learnable gamma
    print("\n" + "=" * 40)
    print("Test 2: Learnable gamma model")
    print("=" * 40)
    
    model_gamma = SpectralCFLearnable(dataset.UserItemNet, config)
    
    # Show gamma parameter
    if hasattr(model_gamma, 'item_gamma'):
        print(f"Initial gamma value: {model_gamma.item_gamma.item():.4f}")
    
    # Show optimizer groups
    print("\nOptimizer groups:")
    for group in model_gamma.get_optimizer_groups():
        print(f"  {group['name']}: lr={group['lr']}, decay={group['weight_decay']}")
    
    # Test forward pass with a few users
    test_users = [0, 1, 2]
    print(f"\nTesting forward pass with users: {test_users}")
    
    # Standard model
    with torch.no_grad():
        scores_standard = model_standard.forward(test_users)
        print(f"Standard model output shape: {scores_standard.shape}")
        print(f"Standard model sample scores: {scores_standard[0, :5]}")
    
    # Learnable gamma model
    with torch.no_grad():
        scores_gamma = model_gamma.forward(test_users)
        print(f"Gamma model output shape: {scores_gamma.shape}")
        print(f"Gamma model sample scores: {scores_gamma[0, :5]}")
    
    # Test gradient computation
    print("\nTesting gradient computation...")
    model_gamma.train()
    scores = model_gamma.forward(test_users)
    loss = scores.sum()  # Dummy loss
    loss.backward()
    
    if hasattr(model_gamma, 'item_gamma'):
        print(f"Gamma gradient: {model_gamma.item_gamma.grad}")
    
    print("\n" + "=" * 60)
    print("Learnable gamma test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_learnable_gamma()