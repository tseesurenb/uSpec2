"""
Efficient Raw Model with Symmetric Softmax - Following DySimGCF Pattern
Works on sparse edges only, not dense matrices
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import time


class EfficientRawModel(nn.Module):
    """
    Efficient symmetric softmax normalization following DySimGCF pattern
    Works on sparse edge structure, not dense matrices
    """
    
    def __init__(self, adj_mat, temperature=1.0):
        super().__init__()
        self.adj_mat = adj_mat
        self.n_users, self.n_items = adj_mat.shape
        self.temperature = temperature
        
        print(f"Efficient Raw Symmetric Softmax: {self.n_users} users, {self.n_items} items")
        
        # Setup sparse edge structure
        self._setup_edges()
    
    def _setup_edges(self):
        """Setup sparse edge structure like GCN implementations"""
        print("Computing item similarity edges...", end=" ", flush=True)
        start = time.time()
        
        # Compute R.T @ R to get item-item edges and weights
        item_sim = self.adj_mat.T @ self.adj_mat
        
        # Convert to COO format to get edge_index and edge_attrs
        item_sim_coo = item_sim.tocoo()
        
        # Create edge_index (2 x num_edges) and edge_attrs (num_edges,)
        self.edge_index = torch.tensor(
            np.vstack([item_sim_coo.row, item_sim_coo.col]), 
            dtype=torch.long
        )
        self.edge_attrs = torch.tensor(item_sim_coo.data, dtype=torch.float32)
        
        # Remove self-loops and zero edges
        mask = (self.edge_index[0] != self.edge_index[1]) & (self.edge_attrs > 1e-10)
        self.edge_index = self.edge_index[:, mask]
        self.edge_attrs = self.edge_attrs[mask]
        
        print(f"✓ ({time.time() - start:.1f}s, {len(self.edge_attrs)} edges)")
        
        # Precompute symmetric softmax normalization
        self._compute_symmetric_softmax()
    
    def _compute_symmetric_softmax(self):
        """Compute symmetric softmax normalization on sparse edges"""
        print("Computing symmetric softmax normalization...", end=" ", flush=True)
        start = time.time()
        
        from_, to_ = self.edge_index
        
        # Apply temperature scaling
        scaled_attrs = self.edge_attrs / self.temperature
        
        # Manual implementation of scatter_softmax
        # Incoming normalization: softmax over edges pointing TO each node
        incoming_norm = torch.zeros_like(scaled_attrs)
        for i in range(self.n_items):
            mask = to_ == i
            if mask.any():
                edge_vals = scaled_attrs[mask]
                softmax_vals = F.softmax(edge_vals, dim=0)
                incoming_norm[mask] = softmax_vals
        
        # Outgoing normalization: softmax over edges pointing FROM each node
        outgoing_norm = torch.zeros_like(scaled_attrs)
        for i in range(self.n_items):
            mask = from_ == i
            if mask.any():
                edge_vals = scaled_attrs[mask]
                softmax_vals = F.softmax(edge_vals, dim=0)
                outgoing_norm[mask] = softmax_vals
        
        # Symmetric normalization: geometric mean
        self.edge_norm = torch.sqrt(incoming_norm * outgoing_norm + 1e-10)
        
        print(f"✓ ({time.time() - start:.1f}s)")
    
    def forward(self, users):
        """Efficient forward pass using sparse operations"""
        # Get user profiles (sparse)
        user_profiles_sparse = self.adj_mat[users]  # (batch_size, n_items)
        user_profiles = torch.tensor(user_profiles_sparse.toarray(), dtype=torch.float32)
        
        # Apply symmetric softmax normalized propagation
        scores = self._propagate(user_profiles)
        
        return scores
    
    def _propagate(self, user_profiles):
        """Propagate using symmetric softmax normalized edges"""
        batch_size = user_profiles.shape[0]
        from_, to_ = self.edge_index
        
        # For each edge, get the normalized weight and aggregate
        scores = torch.zeros_like(user_profiles)
        
        # Aggregate: for each item i, sum from its neighbors j with normalized weights
        for batch_idx in range(batch_size):
            user_vec = user_profiles[batch_idx]  # (n_items,)
            
            # Message passing: scores[i] += sum_j(norm[j->i] * user_vec[j])
            source_values = user_vec[from_] * self.edge_norm  # (num_edges,)
            scores[batch_idx].scatter_add_(0, to_, source_values)
        
        return scores
    
    def getUsersRating(self, batch_users):
        """Interface for evaluation"""
        return self.forward(batch_users).numpy()