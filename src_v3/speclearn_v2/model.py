"""
Simple Raw Model with Symmetric Softmax Normalization
Minimal implementation focused only on replacing degree normalization with learnable softmax
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import time


class RawSymmetricSoftmax(nn.Module):
    """
    Raw item similarity with symmetric softmax normalization
    Replaces D^{-0.5} R D^{-0.5} with learnable softmax normalization
    """
    
    def __init__(self, adj_mat, temperature=1.0):
        super().__init__()
        self.adj_mat = adj_mat
        self.n_users, self.n_items = adj_mat.shape
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        print(f"Raw Symmetric Softmax Model: {self.n_users} users, {self.n_items} items")
        
        # Precompute item similarity
        self._setup_similarity()
    
    def _setup_similarity(self):
        """Setup item similarity with symmetric softmax normalization"""
        start = time.time()
        
        # Compute raw item similarity: R.T @ R
        item_sim = self.adj_mat.T @ self.adj_mat
        item_sim_dense = item_sim.toarray()
        
        # Apply symmetric softmax normalization (replaces degree normalization)
        item_sim_scaled = item_sim_dense / self.temperature.item()
        
        # Row-wise softmax
        row_max = np.max(item_sim_scaled, axis=1, keepdims=True)
        item_sim_exp = np.exp(item_sim_scaled - row_max)
        row_sums = np.sum(item_sim_exp, axis=1, keepdims=True)
        item_sim_softmax = item_sim_exp / (row_sums + 1e-10)
        
        # Store as sparse matrix
        self.item_sim = sp.csr_matrix(item_sim_softmax)
        
        print(f"Setup completed in {time.time() - start:.2f}s")
    
    def forward(self, users):
        """Simple forward pass: user_profiles @ softmax_normalized_similarity"""
        # Get user profiles
        user_profiles = self.adj_mat[users]
        
        # Apply symmetric softmax normalized item similarity
        scores = user_profiles @ self.item_sim
        
        return torch.tensor(scores.toarray(), dtype=torch.float32)
    
    def getUsersRating(self, batch_users):
        """Interface for evaluation"""
        self.eval()
        with torch.no_grad():
            # Recompute if temperature changed
            if hasattr(self, '_last_temp') and self._last_temp != self.temperature.item():
                self._setup_similarity()
            self._last_temp = self.temperature.item()
            
            return self.forward(batch_users).numpy()
    
    def get_optimizer_groups(self):
        """Return temperature parameter for optimization"""
        return [{
            'params': [self.temperature],
            'lr': 0.01,
            'weight_decay': 0.0,
            'name': 'temperature'
        }]