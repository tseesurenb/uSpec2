"""
Minimal Raw Model with Progress Indicators
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import time


class MinimalRawModel(nn.Module):
    """
    Minimal implementation: R.T @ R with symmetric softmax normalization
    """
    
    def __init__(self, adj_mat, temperature=1.0):
        super().__init__()
        self.adj_mat = adj_mat
        self.n_users, self.n_items = adj_mat.shape
        self.temperature = temperature
        
        print(f"Raw Symmetric Softmax: {self.n_users} users, {self.n_items} items")
        
        # Setup similarity with progress
        self._setup_similarity()
    
    def _setup_similarity(self):
        """Setup with progress indicators"""
        print("Computing item similarity (R.T @ R)...", end=" ", flush=True)
        start = time.time()
        
        # Raw item similarity
        item_sim = self.adj_mat.T @ self.adj_mat
        print(f"✓ ({time.time() - start:.1f}s)")
        
        print("Applying symmetric softmax normalization...", end=" ", flush=True)
        start = time.time()
        
        # Convert to dense for softmax
        item_sim_dense = item_sim.toarray()
        
        # Temperature scaling + softmax
        item_sim_scaled = item_sim_dense / self.temperature
        
        # Row-wise softmax
        row_max = np.max(item_sim_scaled, axis=1, keepdims=True)
        item_sim_exp = np.exp(item_sim_scaled - row_max)
        row_sums = np.sum(item_sim_exp, axis=1, keepdims=True)
        item_sim_softmax = item_sim_exp / (row_sums + 1e-10)
        
        # Back to sparse
        self.item_sim = sp.csr_matrix(item_sim_softmax)
        print(f"✓ ({time.time() - start:.1f}s)")
    
    def forward(self, users):
        """Simple forward pass"""
        user_profiles = self.adj_mat[users]
        scores = user_profiles @ self.item_sim
        return torch.tensor(scores.toarray(), dtype=torch.float32)
    
    def getUsersRating(self, batch_users):
        """Interface for evaluation"""
        return self.forward(batch_users).numpy()