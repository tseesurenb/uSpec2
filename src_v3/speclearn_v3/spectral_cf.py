"""
Minimal Spectral CF - Clean implementation inspired by GF-CF
Core: 3-view eigendecomposition with learnable filters
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


class LearnableFilter(nn.Module):
    """Simple learnable spectral filter"""
    def __init__(self, n_eigen, init_type='lowpass'):
        super().__init__()
        self.n_eigen = n_eigen
        self.filter_weights = nn.Parameter(torch.ones(n_eigen))
        
        # Initialize based on type
        with torch.no_grad():
            if init_type == 'lowpass':
                # Low-pass: decay with frequency
                for i in range(n_eigen):
                    self.filter_weights[i] = 0.8 * np.exp(-2 * i / n_eigen)
            elif init_type == 'uniform':
                self.filter_weights.fill_(0.5)
            else:  # 'ones'
                self.filter_weights.fill_(1.0)
    
    def forward(self, eigenvals):
        """Apply learnable filter to eigenvalues"""
        return torch.sigmoid(self.filter_weights)  # Keep in [0,1] range


class SpectralCF(nn.Module):
    """Clean 3-view spectral CF with learnable filters"""
    
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
        
        # Learnable filters for each view
        self.user_filter = LearnableFilter(self.u_eigen, 'lowpass')
        self.item_filter = LearnableFilter(self.i_eigen, 'lowpass') 
        self.bipartite_filter = LearnableFilter(self.b_eigen, 'lowpass')
        
        # Learning rates per view
        self.user_lr = config.get('user_lr', 0.01)
        self.item_lr = config.get('item_lr', 0.01)
        self.bipartite_lr = config.get('bipartite_lr', 0.01)
        
        print(f"Spectral CF: {self.n_users} users, {self.n_items} items")
        print(f"Eigenvalues: u={self.u_eigen}, i={self.i_eigen}, b={self.b_eigen}")
        
        # Setup cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Compute eigendecompositions with caching
        self._setup_eigendecompositions_cached()
    
    def _get_cache_key(self):
        """Generate cache key for similarity matrices"""
        # Create hash of adjacency matrix for unique identification
        adj_hash = hashlib.md5(
            self.adj_mat.data.tobytes() + 
            self.adj_mat.indices.tobytes() + 
            self.adj_mat.indptr.tobytes()
        ).hexdigest()[:12]
        
        return f"sim_{adj_hash}_{self.n_users}_{self.n_items}"
    
    def _setup_eigendecompositions_cached(self):
        """Smart caching: only cache similarity matrices (safe and efficient)"""
        start = time.time()
        cache_key = self._get_cache_key()
        
        # Try to load similarity matrices
        sim_cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(sim_cache_file):
            print("Loading cached similarity matrices...")
            try:
                with open(sim_cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    user_sim = cached['user_sim']
                    item_sim = cached['item_sim'] 
                    bipartite_adj = cached['bipartite_adj']
                print("Loaded similarities from cache")
            except Exception as e:
                print(f"Similarity cache loading failed: {e}, recomputing all...")
                user_sim, item_sim, bipartite_adj = self._compute_similarities()
        else:
            # Compute from scratch
            print("Computing similarity matrices...")
            user_sim, item_sim, bipartite_adj = self._compute_similarities()
            
            # Cache the similarities
            try:
                with open(sim_cache_file, 'wb') as f:
                    pickle.dump({
                        'user_sim': user_sim,
                        'item_sim': item_sim,
                        'bipartite_adj': bipartite_adj
                    }, f)
                print("Cached similarity matrices")
            except Exception as e:
                print(f"Failed to cache similarities: {e}")
        
        # Always compute eigendecompositions fresh (fast and safe)
        print("Computing eigendecompositions...")
        self._compute_eigendecompositions(user_sim, item_sim, bipartite_adj)
        
        print(f"Setup completed in {time.time() - start:.2f}s")
    
    def _compute_similarities(self):
        """Compute the 3 similarity matrices"""
        # GF-CF style normalization
        rowsum = np.array(self.adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt_u = np.power(rowsum + 1e-10, -0.5)
        d_inv_sqrt_u[np.isinf(d_inv_sqrt_u)] = 0.
        d_mat_u = sp.diags(d_inv_sqrt_u)
        
        colsum = np.array(self.adj_mat.sum(axis=0)).flatten()
        d_inv_sqrt_i = np.power(colsum + 1e-10, -0.5)
        d_inv_sqrt_i[np.isinf(d_inv_sqrt_i)] = 0.
        d_mat_i = sp.diags(d_inv_sqrt_i)
        
        # Normalized adjacency
        norm_adj = d_mat_u @ self.adj_mat @ d_mat_i
        
        # Compute similarities
        user_sim = norm_adj @ norm_adj.T
        item_sim = norm_adj.T @ norm_adj
        bipartite_adj = sp.bmat([[None, norm_adj], [norm_adj.T, None]], format='csr')
        
        return user_sim, item_sim, bipartite_adj
    
    def _compute_eigendecompositions(self, user_sim, item_sim, bipartite_adj):
        """Compute eigendecompositions from similarity matrices"""
        # User-user similarity
        u_vals, u_vecs = eigsh(user_sim, k=min(self.u_eigen, user_sim.shape[0]-1), which='LM')
        self.register_buffer('user_eigenvals', torch.tensor(u_vals, dtype=torch.float32))
        self.register_buffer('user_eigenvecs', torch.tensor(u_vecs, dtype=torch.float32))
        
        # Item-item similarity
        i_vals, i_vecs = eigsh(item_sim, k=min(self.i_eigen, item_sim.shape[0]-1), which='LM')
        self.register_buffer('item_eigenvals', torch.tensor(i_vals, dtype=torch.float32))
        self.register_buffer('item_eigenvecs', torch.tensor(i_vecs, dtype=torch.float32))
        
        # Bipartite similarity
        b_vals, b_vecs = eigsh(bipartite_adj, k=min(self.b_eigen, bipartite_adj.shape[0]-1), which='LM')
        self.register_buffer('bipartite_eigenvals', torch.tensor(b_vals, dtype=torch.float32))
        self.register_buffer('bipartite_eigenvecs', torch.tensor(b_vecs, dtype=torch.float32))
        
    def _setup_eigendecompositions(self):
        """Compute eigendecompositions for all 3 views"""
        start = time.time()
        print("Computing eigendecompositions...")
        
        # GF-CF style normalization
        rowsum = np.array(self.adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt_u = np.power(rowsum + 1e-10, -0.5)
        d_inv_sqrt_u[np.isinf(d_inv_sqrt_u)] = 0.
        d_mat_u = sp.diags(d_inv_sqrt_u)
        
        colsum = np.array(self.adj_mat.sum(axis=0)).flatten()
        d_inv_sqrt_i = np.power(colsum + 1e-10, -0.5)
        d_inv_sqrt_i[np.isinf(d_inv_sqrt_i)] = 0.
        d_mat_i = sp.diags(d_inv_sqrt_i)
        
        # Normalized adjacency
        norm_adj = d_mat_u @ self.adj_mat @ d_mat_i
        
        # 1. User-user similarity: R @ R.T
        print("Computing user-user similarity...")
        user_sim = norm_adj @ norm_adj.T
        u_vals, u_vecs = eigsh(user_sim, k=min(self.u_eigen, user_sim.shape[0]-1), which='LM')
        self.register_buffer('user_eigenvals', torch.tensor(u_vals, dtype=torch.float32))
        self.register_buffer('user_eigenvecs', torch.tensor(u_vecs, dtype=torch.float32))
        
        # 2. Item-item similarity: R.T @ R  
        print("Computing item-item similarity...")
        item_sim = norm_adj.T @ norm_adj
        i_vals, i_vecs = eigsh(item_sim, k=min(self.i_eigen, item_sim.shape[0]-1), which='LM')
        self.register_buffer('item_eigenvals', torch.tensor(i_vals, dtype=torch.float32))
        self.register_buffer('item_eigenvecs', torch.tensor(i_vecs, dtype=torch.float32))
        
        # 3. Bipartite adjacency
        print("Computing bipartite similarity...")
        bipartite_adj = sp.bmat([[None, norm_adj], [norm_adj.T, None]], format='csr')
        b_vals, b_vecs = eigsh(bipartite_adj, k=min(self.b_eigen, bipartite_adj.shape[0]-1), which='LM')
        self.register_buffer('bipartite_eigenvals', torch.tensor(b_vals, dtype=torch.float32))
        self.register_buffer('bipartite_eigenvecs', torch.tensor(b_vecs, dtype=torch.float32))
        
        print(f"Eigendecomposition completed in {time.time() - start:.2f}s")
        
    def forward(self, users):
        """Forward pass - compute scores for users"""
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
        
        # User view: filter user similarities
        user_filter_response = self.user_filter(self.user_eigenvals)
        user_vecs = self.user_eigenvecs[users]  # (batch, n_eigen)
        user_filtered = user_vecs @ torch.diag(user_filter_response) @ user_vecs.T @ user_profiles
        scores.append(user_filtered)
        
        # Item view: filter item similarities  
        item_filter_response = self.item_filter(self.item_eigenvals)
        item_filtered = user_profiles @ self.item_eigenvecs @ torch.diag(item_filter_response) @ self.item_eigenvecs.T
        scores.append(item_filtered)
        
        # Bipartite view: filter bipartite similarities
        bipartite_filter_response = self.bipartite_filter(self.bipartite_eigenvals)
        bipartite_vecs = self.bipartite_eigenvecs[users]  # (batch, n_eigen) 
        bipartite_filtered = bipartite_vecs @ torch.diag(bipartite_filter_response) @ bipartite_vecs.T @ user_profiles
        scores.append(bipartite_filtered)
        
        # Simple average of all views
        final_scores = sum(scores) / len(scores)
        return final_scores
    
    def getUsersRating(self, batch_users):
        """Get ratings for evaluation (GF-CF style interface)"""
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


def MSE_train(dataset, model, optimizer, batch_size=1000):
    """Simple MSE training"""
    model.train()
    
    # Sample users
    n_users = dataset.n_users
    sample_size = min(batch_size, n_users)
    users = np.random.choice(n_users, sample_size, replace=False)
    users = torch.tensor(users, dtype=torch.long).to(model.device)
    
    # Create targets (binary: 1 for interactions, 0 for others)
    targets = torch.zeros(len(users), dataset.m_items, device=model.device)
    for i, user in enumerate(users.cpu().numpy()):
        items = dataset.allPos[user]
        if len(items) > 0:
            targets[i, items] = 1.0
    
    # Forward pass
    optimizer.zero_grad()
    predictions = model(users)
    
    # MSE loss
    loss = torch.mean((predictions - targets) ** 2)
    
    # Simple L2 regularization
    reg_loss = 0.001 * sum(torch.norm(p, 2) for p in model.parameters())
    total_loss = loss + reg_loss
    
    # Backward
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()