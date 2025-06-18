#!/usr/bin/env python3
"""
Create a smaller subset of Gowalla for faster hyperparameter tuning
"""

import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix


def create_gowalla_subset(data_path="../../data/gowalla", subset_size=5000, min_interactions=5):
    """
    Create a subset of Gowalla data for faster tuning
    
    Args:
        data_path: Path to gowalla data
        subset_size: Target number of users in subset
        min_interactions: Minimum interactions per user to include
    """
    
    print("Loading original Gowalla data...")
    
    # Load training data
    train_file = os.path.join(data_path, 'train.txt')
    
    user_items = {}
    max_user = 0
    max_item = 0
    
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                user = int(parts[0])
                items = [int(x) for x in parts[1:]]
                user_items[user] = items
                max_user = max(max_user, user)
                max_item = max(max_item, max(items))
    
    print(f"Original data: {len(user_items)} users, {max_item+1} items")
    
    # Filter users with sufficient interactions
    filtered_users = {u: items for u, items in user_items.items() 
                     if len(items) >= min_interactions}
    
    print(f"Users with >={min_interactions} interactions: {len(filtered_users)}")
    
    # Sample subset of users
    if len(filtered_users) > subset_size:
        sampled_users = np.random.choice(list(filtered_users.keys()), 
                                       size=subset_size, replace=False)
        subset_user_items = {u: filtered_users[u] for u in sampled_users}
    else:
        subset_user_items = filtered_users
    
    print(f"Subset: {len(subset_user_items)} users")
    
    # Remap user and item IDs to be continuous
    user_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(subset_user_items.keys()))}
    
    # Get all items in subset
    all_items = set()
    for items in subset_user_items.values():
        all_items.update(items)
    
    item_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(all_items))}
    
    print(f"Remapped to: {len(user_mapping)} users, {len(item_mapping)} items")
    
    # Create subset directory
    subset_dir = "../../data/gowalla_subset"
    os.makedirs(subset_dir, exist_ok=True)
    
    # Split subset into train/test (80/20)
    subset_train = {}
    subset_test = {}
    
    for old_user, items in subset_user_items.items():
        new_user = user_mapping[old_user]
        new_items = [item_mapping[item] for item in items]
        
        # Split items randomly
        np.random.shuffle(new_items)
        split_point = int(len(new_items) * 0.8)
        
        train_items = new_items[:split_point]
        test_items = new_items[split_point:]
        
        if train_items:
            subset_train[new_user] = train_items
        if test_items:
            subset_test[new_user] = test_items
    
    # Write train file
    with open(os.path.join(subset_dir, 'train.txt'), 'w') as f:
        for user in sorted(subset_train.keys()):
            items = subset_train[user]
            f.write(f"{user} {' '.join(map(str, items))}\n")
    
    # Write test file  
    with open(os.path.join(subset_dir, 'test.txt'), 'w') as f:
        for user in sorted(subset_test.keys()):
            items = subset_test[user]
            f.write(f"{user} {' '.join(map(str, items))}\n")
    
    print(f"Subset saved to: {subset_dir}")
    print(f"Train users: {len(subset_train)}, Test users: {len(subset_test)}")
    
    # Calculate sparsity
    total_interactions = sum(len(items) for items in subset_train.values()) + sum(len(items) for items in subset_test.values())
    sparsity = total_interactions / (len(user_mapping) * len(item_mapping))
    print(f"Subset sparsity: {sparsity:.6f}")
    
    return subset_dir


if __name__ == "__main__":
    # Create subset with 5000 users who have at least 5 interactions
    subset_dir = create_gowalla_subset(subset_size=5000, min_interactions=5)
    print(f"Gowalla subset created at: {subset_dir}")