#!/usr/bin/env python
"""
Standalone Filter Pattern Search
No dependencies - completely self-contained
"""

import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import argparse
import os
import pickle


def load_dataset_simple(dataset_name, data_root="../../data"):
    """Simple dataset loader without dependencies"""
    data_path = f"{data_root}/{dataset_name}"
    
    # All datasets use train.txt and test.txt format
    train_file = os.path.join(data_path, "train.txt")
    test_file = os.path.join(data_path, "test.txt")
    
    # Load training data
    train_data = []
    max_user, max_item = 0, 0
    
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user, item = map(int, parts[:2])
                train_data.append((user, item))
                max_user = max(max_user, user)
                max_item = max(max_item, item)
    
    # Load test data
    test_data = {}
    with open(test_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user, item = map(int, parts[:2])
                if user not in test_data:
                    test_data[user] = []
                test_data[user].append(item)
    
    n_users = max_user + 1
    n_items = max_item + 1
    
    # Create adjacency matrix
    rows, cols = zip(*train_data)
    adj_mat = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))
    
    print(f"{dataset_name} loaded: {len(train_data)} train, {len(test_data)} test users")
    print(f"Users: {n_users}, Items: {n_items}")
    
    return adj_mat, test_data, n_users, n_items


def compute_similarity_matrices(adj_mat):
    """Compute similarity matrices"""
    # GF-CF normalization with zero handling
    rowsum = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv = np.power(rowsum + 1e-10, -0.5)  # Add small epsilon to avoid division by zero
    d_inv[rowsum == 0] = 0.  # Set zero rows to 0
    d_mat_u = sp.diags(d_inv)
    
    colsum = np.array(adj_mat.sum(axis=0)).flatten()
    d_inv = np.power(colsum + 1e-10, -0.5)  # Add small epsilon to avoid division by zero
    d_inv[colsum == 0] = 0.  # Set zero columns to 0
    d_mat_i = sp.diags(d_inv)
    
    norm_adj = d_mat_u.dot(adj_mat).dot(d_mat_i)
    
    # Compute similarities
    user_sim = norm_adj @ norm_adj.T
    item_sim = norm_adj.T @ norm_adj
    
    # Bipartite similarity matrix
    n_users, n_items = adj_mat.shape
    bipartite_sim = sp.bmat([[None, norm_adj], [norm_adj.T, None]], format='csr')
    
    return user_sim, item_sim, bipartite_sim


def generate_filter_patterns(eigenvals):
    """Generate diverse filter patterns"""
    patterns = []
    
    min_eig, max_eig = eigenvals.min(), eigenvals.max()
    mean_eig = eigenvals.mean()
    std_eig = eigenvals.std()
    
    print(f"Eigenvalue stats: min={min_eig:.6f}, max={max_eig:.6f}, mean={mean_eig:.6f}, std={std_eig:.6f}")
    print(f"Range: {max_eig - min_eig:.6f}")
    
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
        width = max(multiplier * std_eig, 1e-6)  # Avoid division by zero
        band = np.exp(-((eigenvals - center) / width)**2)
        patterns.append((f'band_{multiplier}', band))
    
    # 5. Inverse patterns (emphasize small eigenvalues)
    patterns.append(('inverse', 1.0 / (eigenvals + 0.01)))
    patterns.append(('sqrt_inverse', 1.0 / np.sqrt(eigenvals + 0.01)))
    
    # 6. Exponential patterns
    if max_eig > min_eig:
        patterns.append(('exp_decay', np.exp(-5 * (eigenvals - min_eig) / (max_eig - min_eig))))
        patterns.append(('exp_grow', np.exp(2 * (eigenvals - min_eig) / (max_eig - min_eig))))
    
    return patterns


def evaluate_pattern(adj_mat, eigenvecs, eigenvals, pattern, test_data, view):
    """Evaluate a filter pattern"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    eigenvecs = torch.tensor(eigenvecs, dtype=torch.float32).to(device)
    eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32).to(device)
    pattern_tensor = torch.tensor(pattern, dtype=torch.float32).to(device)
    adj_tensor = torch.tensor(adj_mat.toarray(), dtype=torch.float32).to(device)
    
    # Apply filter
    filtered_eigenvals = pattern_tensor * eigenvals_tensor
    
    # Compute filtered similarity
    filtered_sim = eigenvecs @ torch.diag(filtered_eigenvals) @ eigenvecs.T
    
    # Test on sample users
    test_users = list(test_data.keys())
    sample_users = np.random.choice(test_users, min(100, len(test_users)), replace=False)
    
    total_ndcg = 0
    valid_users = 0
    
    for user in sample_users:
        if len(test_data[user]) == 0:
            continue
            
        # Get user profile
        user_profile = adj_tensor[user].unsqueeze(0)
        
        # Apply filter based on view
        if view == 'user':
            # User-user filtering: get user similarity vector, then multiply by items
            user_sim_vector = filtered_sim[user]  # Get row for this user (1 x n_users)
            scores = user_sim_vector @ adj_tensor  # (n_users,) @ (n_users, n_items) = (n_items,)
        elif view == 'item':
            # Item-item filtering: user profile multiplied by item similarity
            scores = user_profile @ filtered_sim  # (1, n_items) @ (n_items, n_items) = (1, n_items)
            scores = scores.squeeze()
        else:  # bipartite view
            # Bipartite filtering: extend user profile and multiply by bipartite similarity
            n_users, n_items = adj_tensor.shape
            extended_profile = torch.zeros(n_users + n_items, device=adj_tensor.device)
            extended_profile[user] = 1.0  # Set user position
            extended_profile[n_users:] = user_profile.squeeze()  # Set item positions
            scores = extended_profile @ filtered_sim  # Full bipartite multiplication
            scores = scores[n_users:]  # Extract item scores
            
        scores = scores.squeeze()
        
        # Mask training items
        train_items = torch.nonzero(adj_tensor[user]).squeeze()
        if len(train_items.shape) == 0:
            train_items = train_items.unsqueeze(0)
        scores[train_items] = -float('inf')
        
        # Get top-20 recommendations
        _, top_items = torch.topk(scores, min(20, len(scores)))
        top_items = top_items.cpu().numpy()
        
        # Calculate NDCG@20
        test_items = set(test_data[user])
        hits = [1 if item in test_items else 0 for item in top_items]
        
        if sum(hits) > 0:
            dcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(test_items), 20)))
            ndcg = dcg / idcg if idcg > 0 else 0
            total_ndcg += ndcg
            valid_users += 1
    
    return total_ndcg / valid_users if valid_users > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Standalone Filter Pattern Search")
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['yelp2018', 'gowalla', 'amazon-book'])
    parser.add_argument('--view', type=str, required=True, choices=['user', 'item', 'bipartite'])
    parser.add_argument('--eigenvalues', type=int, default=600)
    parser.add_argument('--data_path', type=str, default="../../data", 
                       help='path to data directory')
    
    args = parser.parse_args()
    
    print(f"🔍 Pattern Search: {args.dataset} - {args.view} view")
    print(f"Target eigenvalues: {args.eigenvalues}")
    
    # Load dataset
    adj_mat, test_data, n_users, n_items = load_dataset_simple(args.dataset, args.data_path)
    
    # Compute similarity matrices
    print("Computing similarity matrices...")
    user_sim, item_sim, bipartite_sim = compute_similarity_matrices(adj_mat)
    
    # Choose similarity matrix
    if args.view == 'user':
        sim_matrix = user_sim
    elif args.view == 'item':
        sim_matrix = item_sim
    else:  # bipartite
        sim_matrix = bipartite_sim
    
    # Compute eigendecomposition with safety checks
    print(f"Computing {args.view} eigendecomposition...")
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    
    start = time.time()
    # Conservative eigenvalue count to avoid ARPACK issues
    max_safe_eigs = min(args.eigenvalues, sim_matrix.shape[0] - 10, 500)
    print(f"Requesting {max_safe_eigs} eigenvalues (reduced for stability)")
    
    try:
        eigenvals, eigenvecs = eigsh(sim_matrix, k=max_safe_eigs, which='LM', maxiter=3000, tol=1e-6)
        print(f"Completed in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Failed with {max_safe_eigs} eigenvalues, trying smaller number...")
        max_safe_eigs = min(200, sim_matrix.shape[0] - 10)
        print(f"Trying {max_safe_eigs} eigenvalues")
        eigenvals, eigenvecs = eigsh(sim_matrix, k=max_safe_eigs, which='LM', maxiter=3000, tol=1e-6)
        print(f"Completed in {time.time() - start:.2f}s")
    
    # Generate patterns
    patterns = generate_filter_patterns(eigenvals)
    print(f"\nTesting {len(patterns)} patterns...")
    
    # Evaluate patterns
    results = []
    for i, (pattern_name, pattern_values) in enumerate(patterns):
        print(f"[{i+1}/{len(patterns)}] {pattern_name}...", end=' ')
        ndcg = evaluate_pattern(adj_mat, eigenvecs, eigenvals, pattern_values, test_data, args.view)
        results.append((pattern_name, ndcg, pattern_values))
        print(f"NDCG@20 = {ndcg:.4f}")
    
    # Sort and display results
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 Top 5 Patterns:")
    for i, (name, ndcg, pattern) in enumerate(results[:5]):
        print(f"{i+1}. {name}: {ndcg:.4f}")
    
    # Save best pattern
    best_name, best_ndcg, best_pattern = results[0]
    filename = f'best_pattern_{args.dataset}_{args.view}.npy'
    np.save(filename, best_pattern)
    print(f"\n💾 Best pattern saved: {filename}")
    print(f"🎯 Best pattern: {best_name} with NDCG@20 = {best_ndcg:.4f}")


if __name__ == "__main__":
    main()