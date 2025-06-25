"""
Laplacian-based Spectral CF - Alternative to similarity matrices
Uses graph Laplacian eigendecomposition for spectral filtering
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import os
import pickle
import hashlib


class LaplacianFilter(nn.Module):
    """Learnable filter for Laplacian spectrum"""
    def __init__(self, n_eigen, init_type='smooth'):
        super().__init__()
        self.n_eigen = n_eigen
        self.filter_weights = nn.Parameter(torch.ones(n_eigen))
        
        # Initialize for Laplacian spectrum (eigenvalues in [0, 2])
        with torch.no_grad():
            if init_type == 'smooth':
                # Low-pass: emphasize small eigenvalues (smooth signals)
                for i in range(n_eigen):
                    self.filter_weights[i] = np.exp(-i / (n_eigen / 4))
            elif init_type == 'highpass':
                # High-pass: emphasize large eigenvalues (varying signals)
                for i in range(n_eigen):
                    self.filter_weights[i] = 1 - np.exp(-i / (n_eigen / 4))
            else:  # 'uniform'
                self.filter_weights.fill_(0.5)
    
    def forward(self, eigenvals):
        """Apply learnable filter to Laplacian eigenvalues"""
        return torch.sigmoid(self.filter_weights)


class LaplacianCF(nn.Module):
    """Laplacian-based Spectral CF using graph Laplacian eigendecomposition"""
    
    def __init__(self, adj_mat, config):
        super().__init__()
        self.device = config.get('device', torch.device('cpu'))
        
        # Convert adjacency matrix
        if sp.issparse(adj_mat):
            self.adj_mat = adj_mat.tocsr()
        else:
            self.adj_mat = sp.csr_matrix(adj_mat)
        
        self.n_users, self.n_items = self.adj_mat.shape
        
        # Eigenvalue configuration
        self.u_eigen = config.get('u_eigen', 64)
        self.i_eigen = config.get('i_eigen', 256) 
        self.b_eigen = config.get('b_eigen', 256)
        
        # Laplacian type
        self.laplacian_type = config.get('laplacian_type', 'normalized')  # 'normalized' or 'random_walk'
        
        # Learnable filters for each Laplacian
        self.user_filter = LaplacianFilter(self.u_eigen, 'smooth')
        self.item_filter = LaplacianFilter(self.i_eigen, 'smooth') 
        self.bipartite_filter = LaplacianFilter(self.b_eigen, 'smooth')
        
        # Learning rates per view
        self.user_lr = config.get('user_lr', 0.01)
        self.item_lr = config.get('item_lr', 0.01)
        self.bipartite_lr = config.get('bipartite_lr', 0.01)
        
        print(f"Laplacian CF: {self.n_users} users, {self.n_items} items")
        print(f"Laplacian type: {self.laplacian_type}")
        print(f"Eigenvalues: u={self.u_eigen}, i={self.i_eigen}, b={self.b_eigen}")
        
        # Setup cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Compute Laplacian eigendecompositions with caching
        self._setup_laplacian_decompositions_cached()
    
    def _get_cache_key(self):
        """Generate cache key for Laplacian matrices"""
        adj_hash = hashlib.md5(
            self.adj_mat.data.tobytes() + 
            self.adj_mat.indices.tobytes() + 
            self.adj_mat.indptr.tobytes()
        ).hexdigest()[:12]
        
        return f"lap_{self.laplacian_type}_{adj_hash}_{self.n_users}_{self.n_items}"
    
    def _setup_laplacian_decompositions_cached(self):
        """Compute and cache Laplacian matrices"""
        start = time.time()
        cache_key = self._get_cache_key()
        
        # Try to load cached Laplacians
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            print("Loading cached Laplacian matrices...")
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    user_lap = cached['user_lap']
                    item_lap = cached['item_lap'] 
                    bipartite_lap = cached['bipartite_lap']
                print("Loaded Laplacians from cache")
            except Exception as e:
                print(f"Cache loading failed: {e}, recomputing...")
                user_lap, item_lap, bipartite_lap = self._compute_laplacians()
        else:
            # Compute from scratch
            print("Computing Laplacian matrices...")
            user_lap, item_lap, bipartite_lap = self._compute_laplacians()
            
            # Cache the Laplacians
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'user_lap': user_lap,
                        'item_lap': item_lap,
                        'bipartite_lap': bipartite_lap
                    }, f)
                print("Cached Laplacian matrices")
            except Exception as e:
                print(f"Failed to cache: {e}")
        
        # Compute eigendecompositions
        print("Computing Laplacian eigendecompositions...")
        self._compute_laplacian_eigendecompositions(user_lap, item_lap, bipartite_lap)
        
        print(f"Setup completed in {time.time() - start:.2f}s")
    
    def _compute_laplacians(self):
        """Compute the 3 Laplacian matrices"""
        # First get similarity matrices (A @ A.T pattern)
        # For user-user graph
        user_adj = self.adj_mat @ self.adj_mat.T  # User adjacency from co-interactions
        
        # For item-item graph  
        item_adj = self.adj_mat.T @ self.adj_mat  # Item adjacency from co-users
        
        # For bipartite graph
        bipartite_adj = sp.bmat([[None, self.adj_mat], [self.adj_mat.T, None]], format='csr')
        
        # Compute Laplacians based on type
        if self.laplacian_type == 'normalized':
            user_lap = self._normalized_laplacian(user_adj)
            item_lap = self._normalized_laplacian(item_adj)
            bipartite_lap = self._normalized_laplacian(bipartite_adj)
        else:  # random_walk
            user_lap = self._random_walk_laplacian(user_adj)
            item_lap = self._random_walk_laplacian(item_adj)
            bipartite_lap = self._random_walk_laplacian(bipartite_adj)
        
        return user_lap, item_lap, bipartite_lap
    
    def _normalized_laplacian(self, adj_matrix):
        """Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)"""
        # Degree matrix
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degrees + 1e-10, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # Normalized adjacency
        norm_adj = d_mat_inv_sqrt @ adj_matrix @ d_mat_inv_sqrt
        
        # Normalized Laplacian
        identity = sp.eye(adj_matrix.shape[0])
        return identity - norm_adj
    
    def _random_walk_laplacian(self, adj_matrix):
        """Compute random walk Laplacian: L = I - D^(-1) A"""
        # Degree matrix
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        d_inv = np.power(degrees + 1e-10, -1.0)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        # Random walk normalized adjacency
        rw_adj = d_mat_inv @ adj_matrix
        
        # Random walk Laplacian
        identity = sp.eye(adj_matrix.shape[0])
        return identity - rw_adj
    
    def _compute_laplacian_eigendecompositions(self, user_lap, item_lap, bipartite_lap):
        """Compute eigendecompositions of Laplacian matrices"""
        # User Laplacian (smallest eigenvalues = smoothest eigenvectors)
        u_vals, u_vecs = eigsh(user_lap, k=min(self.u_eigen, user_lap.shape[0]-1), which='SM')
        self.register_buffer('user_eigenvals', torch.tensor(u_vals, dtype=torch.float32))
        self.register_buffer('user_eigenvecs', torch.tensor(u_vecs, dtype=torch.float32))
        
        # Item Laplacian
        i_vals, i_vecs = eigsh(item_lap, k=min(self.i_eigen, item_lap.shape[0]-1), which='SM')
        self.register_buffer('item_eigenvals', torch.tensor(i_vals, dtype=torch.float32))
        self.register_buffer('item_eigenvecs', torch.tensor(i_vecs, dtype=torch.float32))
        
        # Bipartite Laplacian
        b_vals, b_vecs = eigsh(bipartite_lap, k=min(self.b_eigen, bipartite_lap.shape[0]-1), which='SM')
        self.register_buffer('bipartite_eigenvals', torch.tensor(b_vals, dtype=torch.float32))
        self.register_buffer('bipartite_eigenvecs', torch.tensor(b_vecs, dtype=torch.float32))
        
        print(f"Laplacian eigenvalue ranges:")
        print(f"  User: [{u_vals[0]:.4f}, {u_vals[-1]:.4f}]")
        print(f"  Item: [{i_vals[0]:.4f}, {i_vals[-1]:.4f}]")
        print(f"  Bipartite: [{b_vals[0]:.4f}, {b_vals[-1]:.4f}]")
    
    def forward(self, users):
        """Forward pass - compute scores using Laplacian spectral filtering"""
        if not isinstance(users, torch.Tensor):
            users = torch.tensor(users, dtype=torch.long).to(self.device)
        
        # Get user interaction profiles
        batch_size = len(users)
        user_profiles = torch.zeros(batch_size, self.n_items, device=self.device)
        for i, user_id in enumerate(users):
            items = self.adj_mat[user_id.item()].indices
            if len(items) > 0:
                user_profiles[i, items] = 1.0
        
        scores = []
        
        # User Laplacian filtering (graph smoothing)
        user_filter_response = self.user_filter(self.user_eigenvals)
        user_vecs = self.user_eigenvecs[users]  # (batch, n_eigen)
        # Apply filter: reconstruct with filtered eigenvalues
        user_filtered = user_vecs @ torch.diag(user_filter_response) @ user_vecs.T @ user_profiles
        scores.append(user_filtered)
        
        # Item Laplacian filtering
        item_filter_response = self.item_filter(self.item_eigenvals)
        # For items, we need to project and reconstruct
        item_filtered = user_profiles @ self.item_eigenvecs @ torch.diag(item_filter_response) @ self.item_eigenvecs.T
        scores.append(item_filtered)
        
        # Bipartite Laplacian filtering
        bipartite_filter_response = self.bipartite_filter(self.bipartite_eigenvals)
        bipartite_vecs = self.bipartite_eigenvecs[users]  # (batch, n_eigen) 
        bipartite_filtered = bipartite_vecs @ torch.diag(bipartite_filter_response) @ bipartite_vecs.T @ user_profiles
        scores.append(bipartite_filtered)
        
        # Combine all filtered signals
        final_scores = sum(scores) / len(scores)
        return final_scores
    
    def getUsersRating(self, batch_users):
        """Get ratings for evaluation"""
        self.eval()
        with torch.no_grad():
            scores = self.forward(batch_users)
            return scores.cpu().numpy()
    
    def get_optimizer_groups(self):
        """Per-view learning rates"""
        return [
            {'params': self.user_filter.parameters(), 'lr': self.user_lr, 'name': 'user_filter'},
            {'params': self.item_filter.parameters(), 'lr': self.item_lr, 'name': 'item_filter'},
            {'params': self.bipartite_filter.parameters(), 'lr': self.bipartite_lr, 'name': 'bipartite_filter'}
        ]