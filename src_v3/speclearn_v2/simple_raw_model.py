"""
Simple Raw Model with Symmetric Softmax - Minimal and Fast like GF-CF
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import time


class SimpleRawModel(nn.Module):
    """
    Minimal implementation matching GF-CF's efficiency
    Only difference: symmetric softmax normalization instead of degree normalization
    """
    
    def __init__(self, adj_mat, config):
        super().__init__()
        self.adj_mat = adj_mat
        self.n_users, self.n_items = adj_mat.shape
        self.device = config.get('device', torch.device('cpu'))
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(config.get('temperature', 1.0)))
        
        print(f"Simple Raw Model: {self.n_users} users, {self.n_items} items")
        
        # Precompute normalized adjacency (like GF-CF but with softmax)
        self._setup_normalization()
    
    def _setup_normalization(self):
        """Setup symmetric softmax normalization"""
        start = time.time()
        
        # Compute R.T @ R for item similarity (sparse)
        item_sim = self.adj_mat.T @ self.adj_mat
        
        # Convert to dense for softmax (only for items, much smaller than user-item)
        item_sim_dense = item_sim.toarray()
        
        # Apply temperature and softmax normalization
        # This replaces D^{-0.5} R D^{-0.5} with learnable softmax
        item_sim_dense = item_sim_dense / self.temperature.item()
        
        # Row-wise softmax normalization
        row_max = np.max(item_sim_dense, axis=1, keepdims=True)
        item_sim_dense = np.exp(item_sim_dense - row_max)
        row_sums = np.sum(item_sim_dense, axis=1, keepdims=True)
        item_sim_dense = item_sim_dense / (row_sums + 1e-10)
        
        # Convert back to sparse for efficiency
        self.norm_item_sim = sp.csr_matrix(item_sim_dense)
        
        print(f"Setup completed in {time.time() - start:.2f}s")
    
    def forward(self, users):
        """Simple forward pass like GF-CF"""
        # Get batch user profiles (sparse)
        batch_test = self.adj_mat[users]
        
        # Two-hop propagation with symmetric softmax normalized similarity
        # Equivalent to: user_profile @ item_sim_normalized
        scores = batch_test @ self.norm_item_sim
        
        # Convert to dense tensor for PyTorch
        scores = torch.tensor(scores.toarray(), dtype=torch.float32, device=self.device)
        
        return scores
    
    def getUsersRating(self, batch_users):
        """Interface compatibility"""
        self.eval()
        with torch.no_grad():
            # Recompute normalization if temperature changed
            if hasattr(self, '_last_temp') and self._last_temp != self.temperature.item():
                self._setup_normalization()
            self._last_temp = self.temperature.item()
            
            return self.forward(batch_users).cpu().numpy()
    
    def get_optimizer_groups(self):
        """Get parameter groups for optimization"""
        return [{
            'params': [self.temperature],
            'lr': 0.01,
            'weight_decay': 0.0,
            'name': 'temperature'
        }]