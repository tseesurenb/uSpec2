"""
Improved Raw Item Similarity Model with DySimGCF Enhancements
Combines raw two-hop propagation with symmetric softmax normalization and temperature amplification
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import time
import os
import hashlib
import pickle

class ImprovedRawItemSim(nn.Module):
    """
    Enhanced raw item similarity model incorporating DySimGCF ideas:
    - Symmetric softmax normalization with similarity scores as attention
    - Temperature-based similarity amplification
    - Top-K similarity selection for noise reduction
    - Raw two-hop propagation (no spectral filtering)
    """
    
    def __init__(self, adj_mat, config):
        super().__init__()
        self.adj_mat = adj_mat
        self.n_users, self.n_items = adj_mat.shape
        self.device = config.get('device', torch.device('cpu'))
        self.dataset = config.get('dataset', 'unknown')
        
        # DySimGCF enhancements
        self.temperature = nn.Parameter(torch.tensor(config.get('temperature', 1.0)))
        self.top_k_items = config.get('top_k_items', 50)  # Top-K similar items
        
        # Cache directory
        self.cache_dir = config.get('cache_dir', './cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"Improved Raw Item Similarity Model: {self.n_users} users, {self.n_items} items")
        print(f"Temperature: {self.temperature.item():.3f}, Top-K items: {self.top_k_items}")
        print(f"Using raw matrix multiplication (R.T @ R) with symmetric softmax normalization")
        
        # Setup raw similarity without degree normalization
        self._setup_raw_similarity()
        
        # Convert adjacency to tensor
        self.register_buffer('adj_tensor', torch.tensor(adj_mat.toarray(), dtype=torch.float32))
    
    def get_cache_key(self):
        """Generate cache key for similarity matrices"""
        adj_hash = hashlib.md5(
            self.adj_mat.data.tobytes() + 
            self.adj_mat.indices.tobytes() + 
            self.adj_mat.indptr.tobytes()
        ).hexdigest()
        return f"raw_sim_{self.n_users}_{self.n_items}_{self.top_k_items}_{adj_hash[:16]}"
    
    def _compute_raw_similarity(self):
        """Compute raw item similarity using R.T @ R (no degree normalization)"""
        print("Computing raw item similarity (R.T @ R)...")
        
        # Convert adjacency to tensor for GPU computation
        adj_tensor = torch.tensor(self.adj_mat.toarray(), dtype=torch.float32).to(self.device)
        
        # Raw item similarity: R.T @ R (no normalization)
        raw_sim = torch.mm(adj_tensor.T, adj_tensor)
        
        return adj_tensor, raw_sim
    
    def _setup_raw_similarity(self):
        """Setup raw similarity using R.T @ R with caching (no degree normalization)"""
        start = time.time()
        
        # Try loading cached similarity
        cache_key = self.get_cache_key()
        cache_file = os.path.join(self.cache_dir, f"raw_sim_{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            print(f"Loading cached raw similarity matrix...")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                adj_tensor = cached['adj_tensor']
                raw_sim_values = cached['raw_sim_values']
                top_k_indices = cached['top_k_indices']
                top_k_values = cached['top_k_values']
        else:
            print(f"Computing raw similarity matrices...")
            
            # Compute raw similarity without degree normalization
            adj_tensor, raw_sim = self._compute_raw_similarity()
            raw_sim_values = raw_sim.cpu().numpy()
            
            # Apply top-K selection for each item
            print(f"Applying top-{self.top_k_items} selection...")
            top_k_indices = np.zeros((self.n_items, self.top_k_items), dtype=np.int32)
            top_k_values = np.zeros((self.n_items, self.top_k_items), dtype=np.float32)
            
            for i in range(self.n_items):
                # Get top-K items with highest edge weights (excluding self)
                edge_weights = raw_sim_values[i]
                edge_weights[i] = -1  # Exclude self
                
                # Get top-K indices
                top_indices = np.argpartition(edge_weights, -self.top_k_items)[-self.top_k_items:]
                top_indices = top_indices[np.argsort(edge_weights[top_indices])[::-1]]
                
                top_k_indices[i] = top_indices
                top_k_values[i] = edge_weights[top_indices]
            
            # Cache results
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'adj_tensor': adj_tensor.cpu().numpy(),
                        'raw_sim_values': raw_sim_values,
                        'top_k_indices': top_k_indices,
                        'top_k_values': top_k_values,
                        'n_users': self.n_users,
                        'n_items': self.n_items,
                        'top_k_items': self.top_k_items
                    }, f)
                print(f"Cached raw similarity matrices")
            except Exception as e:
                print(f"Failed to cache: {e}")
        
        # Convert to tensors
        self.register_buffer('adj_matrix', torch.tensor(adj_tensor if isinstance(adj_tensor, np.ndarray) else adj_tensor.cpu().numpy(), dtype=torch.float32))
        self.register_buffer('top_k_indices', torch.tensor(top_k_indices, dtype=torch.long))
        self.register_buffer('top_k_values', torch.tensor(top_k_values, dtype=torch.float32))
        
        print(f"Raw similarity setup completed in {time.time() - start:.2f}s")
    
    def _compute_symmetric_softmax(self):
        """
        Compute symmetric softmax normalization - vectorized for speed
        """
        # Apply temperature scaling to raw edge weights
        temp_scaled_values = self.top_k_values / self.temperature
        
        # Vectorized computation of edge weight sums for all items
        edge_weight_sums = torch.sum(torch.exp(temp_scaled_values), dim=1)  # (n_items,)
        
        # Get neighbor sums using advanced indexing (vectorized)
        neighbor_sums = edge_weight_sums[self.top_k_indices]  # (n_items, top_k_items)
        
        # Vectorized symmetric normalization
        exp_weights = torch.exp(temp_scaled_values)  # (n_items, top_k_items)
        norm_factors = torch.sqrt(edge_weight_sums.unsqueeze(1) * neighbor_sums + 1e-10)
        attention_weights = exp_weights / norm_factors
        
        return attention_weights
    
    def forward(self, users):
        """Forward pass using symmetric softmax normalization - efficient version"""
        if not isinstance(users, torch.Tensor):
            users = torch.tensor(users, dtype=torch.long).to(self.device)
        
        # Get user profiles from raw adjacency matrix
        user_profiles = self.adj_matrix[users]  # (batch_size, n_items)
        
        # Compute symmetric softmax attention weights (cache invalidated when temperature changes)
        if not hasattr(self, '_cached_attention_weights') or not hasattr(self, '_cached_temperature') or self._cached_temperature != self.temperature.item():
            self._cached_attention_weights = self._compute_symmetric_softmax()
            self._cached_temperature = self.temperature.item()
        
        # Efficient forward pass without creating full matrix
        batch_size = user_profiles.shape[0]
        scores = torch.zeros(batch_size, self.n_items, device=self.device)
        
        # Batch computation for efficiency
        batch_size_items = 5000  # Process items in chunks
        for start_idx in range(0, self.n_items, batch_size_items):
            end_idx = min(start_idx + batch_size_items, self.n_items)
            
            # Get indices and weights for this batch of items
            batch_neighbors = self.top_k_indices[start_idx:end_idx]  # (batch_items, top_k)
            batch_weights = self._cached_attention_weights[start_idx:end_idx]  # (batch_items, top_k)
            
            # Gather neighbor profiles for all items in batch: (batch_size, batch_items, top_k)
            neighbor_profiles = user_profiles[:, batch_neighbors.reshape(-1)].reshape(
                batch_size, end_idx - start_idx, self.top_k_items
            )
            
            # Compute scores for this batch: (batch_size, batch_items)
            batch_scores = torch.sum(neighbor_profiles * batch_weights.unsqueeze(0), dim=2)
            scores[:, start_idx:end_idx] = batch_scores
        
        return scores
    
    def getUsersRating(self, batch_users):
        """Get ratings for evaluation (interface compatibility)"""
        self.eval()
        with torch.no_grad():
            scores = self.forward(batch_users)
            return scores.cpu().numpy()
    
    def get_optimizer_groups(self):
        """Get parameter groups for optimization"""
        groups = [{
            'params': [self.temperature],
            'lr': 0.01,
            'weight_decay': 0.0,
            'name': 'temperature'
        }]
        return groups