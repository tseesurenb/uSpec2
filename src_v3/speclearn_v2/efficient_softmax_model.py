"""
Efficient Raw Model with Sparse Softmax Normalization
Maintains sparsity while applying softmax normalization
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import time


class EfficientRawSoftmax(nn.Module):
    """
    Raw item similarity with efficient sparse softmax normalization
    """
    
    def __init__(self, adj_mat, temperature=1.0):
        super().__init__()
        self.adj_mat = adj_mat
        self.n_users, self.n_items = adj_mat.shape
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        print(f"Efficient Raw Softmax Model: {self.n_users} users, {self.n_items} items")
        
        # Precompute item similarity
        self._setup_similarity()
    
    def _setup_similarity(self):
        """Setup item similarity with sparse softmax normalization"""
        start = time.time()
        
        # Compute raw item similarity: R.T @ R (stays sparse)
        item_sim = self.adj_mat.T @ self.adj_mat
        
        # Apply sparse softmax normalization row by row
        print("Applying sparse softmax normalization...")
        
        # Convert to lil_matrix for efficient row access
        item_sim_lil = item_sim.tolil()
        
        # Process each row
        for i in range(self.n_items):
            if i % 5000 == 0:
                print(f"Processing row {i}/{self.n_items}")
            
            # Get non-zero indices and values for this row
            row_data = item_sim_lil.data[i]
            if len(row_data) == 0:
                continue
            
            # Apply temperature scaling and softmax only to non-zero entries
            row_array = np.array(row_data)
            row_scaled = row_array / self.temperature.item()
            row_max = np.max(row_scaled)
            row_exp = np.exp(row_scaled - row_max)
            row_sum = np.sum(row_exp)
            
            # Update with normalized values
            item_sim_lil.data[i] = (row_exp / row_sum).tolist()
        
        # Convert back to csr for efficient matrix multiplication
        self.item_sim = item_sim_lil.tocsr()
        
        print(f"Setup completed in {time.time() - start:.2f}s")
        print(f"Sparsity: {self.item_sim.nnz} / {self.n_items**2} = {self.item_sim.nnz/self.n_items**2:.4%}")
    
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