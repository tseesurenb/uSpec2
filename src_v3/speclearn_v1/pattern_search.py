#!/usr/bin/env python
"""
Filter Pattern Search for Learnable Spectral CF
Find best static filter patterns for initialization
Tests various patterns on clustered eigenvalues (crucial for Amazon-book, Yelp, etc.)
"""

import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import argparse

# Import from current directory
from dataloader import Loader


def compute_similarity_matrices(adj_mat):
    """Compute similarity matrices like in static model"""
    # GF-CF normalization
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_u = sp.diags(d_inv)
    
    colsum = np.array(adj_mat.sum(axis=0))
    d_inv = np.power(colsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_i = sp.diags(d_inv)
    
    norm_adj = d_mat_u.dot(adj_mat).dot(d_mat_i)
    
    # Compute similarities
    user_sim = norm_adj @ norm_adj.T
    item_sim = norm_adj.T @ norm_adj
    
    return user_sim, item_sim, norm_adj


def generate_filter_patterns(eigenvals, n_patterns=20):
    """Generate diverse filter patterns"""
    patterns = []
    
    min_eig, max_eig = eigenvals.min(), eigenvals.max()
    mean_eig = eigenvals.mean()
    std_eig = eigenvals.std()
    
    print(f"Eigenvalue stats: min={min_eig:.4f}, max={max_eig:.4f}, mean={mean_eig:.4f}, std={std_eig:.4f}")
    
    # 1. Constant patterns
    patterns.append(('constant_1', np.ones_like(eigenvals)))
    patterns.append(('constant_0.5', 0.5 * np.ones_like(eigenvals)))
    patterns.append(('constant_0.1', 0.1 * np.ones_like(eigenvals)))
    
    # 2. Linear patterns
    patterns.append(('linear_inc', np.linspace(0.1, 1.0, len(eigenvals))))
    patterns.append(('linear_dec', np.linspace(1.0, 0.1, len(eigenvals))))
    
    # 3. Step functions (for clustered eigenvalues)
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        step_filter = np.where(eigenvals > mean_eig + threshold * std_eig, 1.0, 0.1)
        patterns.append((f'step_{threshold}', step_filter))
    
    # 4. Band-pass around eigenvalue clusters
    for multiplier in [0.5, 1.0, 1.5, 2.0]:
        center = mean_eig
        width = multiplier * std_eig
        band = np.exp(-((eigenvals - center) / width)**2)
        patterns.append((f'band_{multiplier}', band))
    
    # 5. Inverse patterns (emphasize small eigenvalues)
    patterns.append(('inverse', 1.0 / (eigenvals + 0.01)))
    patterns.append(('sqrt_inverse', 1.0 / np.sqrt(eigenvals + 0.01)))
    
    # 6. Exponential patterns
    patterns.append(('exp_decay', np.exp(-5 * (eigenvals - min_eig) / (max_eig - min_eig))))
    patterns.append(('exp_grow', np.exp(2 * (eigenvals - min_eig) / (max_eig - min_eig))))
    
    return patterns


def evaluate_static_pattern(adj_tensor, eigenvecs, eigenvals, pattern, users_test, items_test):
    """Evaluate a static filter pattern"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    eigenvecs = torch.tensor(eigenvecs, dtype=torch.float32).to(device)
    eigenvals = torch.tensor(eigenvals, dtype=torch.float32).to(device)
    pattern = torch.tensor(pattern, dtype=torch.float32).to(device)
    adj_tensor = torch.tensor(adj_tensor, dtype=torch.float32).to(device)
    
    # Apply filter
    filtered_eigenvals = pattern * eigenvals
    
    # Compute filtered similarity
    filtered_sim = eigenvecs @ torch.diag(filtered_eigenvals) @ eigenvecs.T
    
    # Test on a sample of users
    sample_users = np.random.choice(len(users_test), min(100, len(users_test)), replace=False)
    
    total_ndcg = 0
    valid_users = 0
    
    for user_idx in sample_users:
        user = users_test[user_idx]
        if len(items_test[user_idx]) == 0:
            continue
            
        # Get user profile
        user_profile = adj_tensor[user].unsqueeze(0)  # (1, n_items)
        
        # Apply filtered similarity
        scores = user_profile @ filtered_sim  # User filtering
        scores = scores.squeeze()
        
        # Mask training items
        train_items = torch.nonzero(adj_tensor[user]).squeeze()
        if len(train_items.shape) == 0:
            train_items = train_items.unsqueeze(0)
        scores[train_items] = -float('inf')
        
        # Get top-k recommendations
        _, top_items = torch.topk(scores, min(20, len(scores)))
        top_items = top_items.cpu().numpy()
        
        # Calculate NDCG@20
        test_items = set(items_test[user_idx])
        hits = [1 if item in test_items else 0 for item in top_items]
        
        if sum(hits) > 0:
            dcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(test_items), 20)))
            ndcg = dcg / idcg if idcg > 0 else 0
            total_ndcg += ndcg
            valid_users += 1
    
    return total_ndcg / valid_users if valid_users > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Filter Pattern Search")
    parser.add_argument('--dataset', type=str, default='amazon-book',
                       choices=['ml-100k', 'yelp2018', 'gowalla', 'amazon-book'])
    parser.add_argument('--view', type=str, default='item', choices=['user', 'item'])
    parser.add_argument('--eigenvalues', type=int, default=600)
    
    args = parser.parse_args()
    
    print(f"🔍 Filter Pattern Search: {args.dataset} - {args.view} view")
    
    # Basic config
    config = {
        'dataset': args.dataset,
        'full_training': True,
        'seed': 2020
    }
    
    # Load dataset
    dataset = Loader(config)
    
    # Get adjacency matrix
    if hasattr(dataset, 'UserItemNet'):
        adj_mat = dataset.UserItemNet
    else:
        # Create adjacency matrix from dataset
        from scipy.sparse import csr_matrix
        rows, cols = [], []
        for user in range(dataset.n_users):
            items = dataset.allPos[user]
            rows.extend([user] * len(items))
            cols.extend(items)
        adj_mat = csr_matrix((np.ones(len(rows)), (rows, cols)), 
                           shape=(dataset.n_users, dataset.m_items))
    
    # Compute similarity matrices
    print("Computing similarity matrices...")
    user_sim, item_sim, norm_adj = compute_similarity_matrices(adj_mat)
    
    # Choose which similarity to use
    if args.view == 'user':
        sim_matrix = user_sim
    else:
        sim_matrix = item_sim
    
    # Compute eigendecomposition
    print(f"Computing eigendecomposition for {args.view} view...")
    start = time.time()
    eigenvals, eigenvecs = eigsh(sim_matrix, k=min(args.eigenvalues, sim_matrix.shape[0]-1), which='LM')
    print(f"Eigendecomposition completed in {time.time() - start:.2f}s")
    
    # Get test data
    users_test = []
    items_test = []
    for user in dataset.testDict:
        users_test.append(user)
        items_test.append(dataset.testDict[user])
    
    # Generate and test patterns
    patterns = generate_filter_patterns(eigenvals)
    print(f"\nTesting {len(patterns)} patterns...")
    
    results = []
    for pattern_name, pattern_values in patterns:
        print(f"Testing {pattern_name}...", end=' ')
        ndcg = evaluate_static_pattern(
            adj_mat.toarray() if hasattr(adj_mat, 'toarray') else adj_mat,
            eigenvecs, eigenvals, pattern_values, users_test, items_test
        )
        results.append((pattern_name, ndcg, pattern_values))
        print(f"NDCG@20 = {ndcg:.4f}")
    
    # Sort by performance
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 Top 5 Patterns:")
    for i, (name, ndcg, pattern) in enumerate(results[:5]):
        print(f"{i+1}. {name}: {ndcg:.4f}")
        print(f"   Pattern sample: {pattern[:5]} ... {pattern[-5:]}")
    
    # Save best pattern for learnable initialization
    best_name, best_ndcg, best_pattern = results[0]
    np.save(f'best_pattern_{args.dataset}_{args.view}.npy', best_pattern)
    print(f"\n💾 Best pattern saved to best_pattern_{args.dataset}_{args.view}.npy")
    print(f"Use this pattern to initialize your learnable {args.view} filter!")


if __name__ == "__main__":
    main()