"""
Learnable Spectral Collaborative Filtering Model
Clean implementation with per-view learning rates
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
from learnable_filters import LearnableSpectralFilter


class SpectralCFLearnable(nn.Module):
    """Learnable Spectral CF with separate filters per view"""
    
    def __init__(self, adj_mat, config):
        super().__init__()
        
        # Basic setup
        self.config = config
        self.device = config.get('device', torch.device('cpu'))
        self.dataset = config.get('dataset', 'unknown')
        
        # Which views to use
        self.filter_views = config.get('filter', 'uib')  # u, i, b, ui, ub, ib, uib
        
        # Convert adjacency matrix
        if sp.issparse(adj_mat):
            self.adj_mat = adj_mat.tocsr()
            adj_dense = adj_mat.toarray()
        else:
            self.adj_mat = sp.csr_matrix(adj_mat)
            adj_dense = adj_mat
            
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32).to(self.device))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Cache setup
        self.cache_dir = os.path.join(os.path.dirname(__file__), '../cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Eigenvalue configuration
        self.u_n_eigen = config.get('u_n_eigen', 25)
        self.i_n_eigen = config.get('i_n_eigen', 200) 
        self.b_n_eigen = config.get('b_n_eigen', 220)
        
        # Filter configuration
        filter_type = config.get('filter_type', 'bernstein')
        filter_order = config.get('filter_order', 8)
        
        # Create learnable filters for each active view
        if 'u' in self.filter_views:
            self.user_filter = LearnableSpectralFilter(
                filter_type=filter_type,
                order=filter_order,
                init_type=config.get('user_init', 'smooth')
            )
        
        if 'i' in self.filter_views:
            self.item_filter = LearnableSpectralFilter(
                filter_type=filter_type,
                order=filter_order,
                init_type=config.get('item_init', 'sharp')
            )
        
        if 'b' in self.filter_views:
            self.bipartite_filter = LearnableSpectralFilter(
                filter_type=filter_type,
                order=filter_order,
                init_type=config.get('bipartite_init', 'smooth')
            )
        
        # View-specific hyperparameters
        self.user_lr = config.get('user_lr', 0.1)
        self.item_lr = config.get('item_lr', 0.01)
        self.bipartite_lr = config.get('bipartite_lr', 0.05)
        
        self.user_decay = config.get('user_decay', 1e-4)
        self.item_decay = config.get('item_decay', 1e-3)
        self.bipartite_decay = config.get('bipartite_decay', 5e-4)
        
        # Two-hop propagation configuration
        self.use_two_hop = config.get('use_two_hop', False)
        self.raw_only = config.get('raw_only', False)  # New: raw propagation only mode
        self.two_hop_weight = nn.Parameter(torch.tensor(config.get('two_hop_weight', 0.3)))
        
        # Learnable normalization parameter for item view (PolyCF inspired)
        self.learnable_gamma = config.get('learnable_gamma', True)
        if self.learnable_gamma and 'i' in self.filter_views:
            # Initialize gamma to 0.5 (standard GF-CF normalization)
            self.item_gamma = nn.Parameter(torch.tensor(0.5))
            print(f"Learnable gamma enabled for item view (init: 0.5)")
        
        # Precompute normalized adjacency for two-hop if needed
        if self.use_two_hop or self.raw_only:
            self._setup_two_hop_matrices()
        
        print(f"SpectralCF Learnable: {self.n_users} users, {self.n_items} items")
        print(f"Active views: {self.filter_views}")
        print(f"Filter type: {filter_type}, order: {filter_order}")
        print(f"Eigenvalues: u={self.u_n_eigen}, i={self.i_n_eigen}, b={self.b_n_eigen}")
        
        # Store raw matrices for learnable gamma
        if self.learnable_gamma and 'i' in self.filter_views:
            self._store_raw_matrices()
        
        # Compute eigendecompositions (skip if raw_only mode)
        if not self.raw_only:
            self._setup_spectral_filters()
    
    def _store_raw_matrices(self):
        """Store raw degree matrices for learnable gamma computation"""
        print("Storing raw matrices for learnable gamma...")
        
        # User degrees
        rowsum = np.array(self.adj_mat.sum(axis=1))
        self.user_degrees = torch.tensor(rowsum.flatten(), dtype=torch.float32).to(self.device)
        
        # Item degrees  
        colsum = np.array(self.adj_mat.sum(axis=0))
        self.item_degrees = torch.tensor(colsum.flatten(), dtype=torch.float32).to(self.device)
        
        print("Raw matrices stored")
    
    def _compute_item_eigenvectors_dynamic(self):
        """Compute item eigenvectors with current gamma value"""
        gamma = self.item_gamma
        
        # Apply learnable normalization: D_u^{-γ} R D_i^{γ-1}
        user_norm = torch.pow(self.user_degrees + 1e-10, -gamma).unsqueeze(1)  # (n_users, 1)
        item_norm = torch.pow(self.item_degrees + 1e-10, gamma - 1.0).unsqueeze(0)  # (1, n_items)
        
        # Normalize adjacency matrix
        norm_adj = self.adj_tensor * user_norm * item_norm  # Broadcasting
        
        # Compute item-item similarity: norm_adj.T @ norm_adj
        item_sim = norm_adj.T @ norm_adj  # (n_items, n_items)
        
        # Eigendecomposition
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(item_sim)
            # Sort by eigenvalue magnitude (descending)
            idx = torch.argsort(torch.abs(eigenvals), descending=True)
            eigenvals = eigenvals[idx[:self.i_n_eigen]]
            eigenvecs = eigenvecs[:, idx[:self.i_n_eigen]]
        except:
            # Fallback to existing eigenvectors if computation fails
            eigenvals = self.item_eigenvals
            eigenvecs = self.item_eigenvecs
        
        return eigenvecs, eigenvals
    
    def get_cache_key(self):
        """Generate cache key for similarity matrices"""
        adj_hash = hashlib.md5(
            self.adj_mat.data.tobytes() + 
            self.adj_mat.indices.tobytes() + 
            self.adj_mat.indptr.tobytes()
        ).hexdigest()
        
        # Include configuration that affects similarity computation
        gamma_str = ""
        if self.learnable_gamma and hasattr(self, 'item_gamma'):
            gamma_str = f"_gamma{self.item_gamma.item():.3f}"
        
        # Include active views to avoid loading wrong matrices
        views_str = f"_views{self.filter_views}"
        
        return f"gfcf_{self.n_users}_{self.n_items}_{adj_hash[:16]}{views_str}{gamma_str}"
    
    def _setup_spectral_filters(self):
        """Compute eigendecompositions for active views"""
        start = time.time()
        
        # Try loading cached similarities (always enabled for similarity matrices)
        cache_key = self.get_cache_key()
        cache_file = os.path.join(self.cache_dir, f"similarities_{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            print(f"Loading cached similarity matrices...")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                user_sim = cached.get('user_sim')
                item_sim = cached.get('item_sim')
                bipartite_sim = cached.get('bipartite_sim')
        else:
            print(f"Computing similarity matrices...")
            user_sim = None
            item_sim = None
            bipartite_sim = None
            
            if 'u' in self.filter_views:
                user_sim = self._compute_user_similarity()
            if 'i' in self.filter_views:
                item_sim = self._compute_item_similarity()
            if 'b' in self.filter_views:
                bipartite_sim = self._compute_bipartite_similarity()
            
            # Always cache similarities
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'user_sim': user_sim,
                            'item_sim': item_sim,
                            'bipartite_sim': bipartite_sim,
                            'n_users': self.n_users,
                            'n_items': self.n_items
                        }, f)
                    print(f"Cached similarity matrices")
                except Exception as e:
                    print(f"Failed to cache: {e}")
        
        # Compute eigendecompositions
        print("Computing eigendecompositions...")
        
        if 'u' in self.filter_views and user_sim is not None:
            eigenvals, eigenvecs = eigsh(user_sim, k=min(self.u_n_eigen, user_sim.shape[0]-1), which='LM')
            self.register_buffer('user_eigenvals', torch.tensor(eigenvals, dtype=torch.float32).to(self.device))
            self.register_buffer('user_eigenvecs', torch.tensor(eigenvecs, dtype=torch.float32).to(self.device))
            print(f"\nUser eigenvals ({len(self.user_eigenvals)} total):")
            print(self.user_eigenvals.numpy())
        
        if 'i' in self.filter_views and item_sim is not None:
            eigenvals, eigenvecs = eigsh(item_sim, k=min(self.i_n_eigen, item_sim.shape[0]-1), which='LM')
            self.register_buffer('item_eigenvals', torch.tensor(eigenvals, dtype=torch.float32).to(self.device))
            self.register_buffer('item_eigenvecs', torch.tensor(eigenvecs, dtype=torch.float32).to(self.device))
            print(f"\nItem eigenvals ({len(self.item_eigenvals)} total):")
            print(self.item_eigenvals.numpy())
        
        if 'b' in self.filter_views and bipartite_sim is not None:
            eigenvals, eigenvecs = eigsh(bipartite_sim, k=min(self.b_n_eigen, bipartite_sim.shape[0]-1), which='LM')
            self.register_buffer('bipartite_eigenvals', torch.tensor(eigenvals, dtype=torch.float32).to(self.device))
            self.register_buffer('bipartite_eigenvecs', torch.tensor(eigenvecs, dtype=torch.float32).to(self.device))
            print(f"\nBipartite eigenvals ({len(self.bipartite_eigenvals)} total):")
            print(self.bipartite_eigenvals.numpy())
        
        print(f"Setup completed in {time.time() - start:.2f}s")
    
    def _setup_two_hop_matrices(self):
        """Setup normalized adjacency matrices for two-hop propagation"""
        print("Setting up two-hop propagation matrices...")
        
        # Compute normalized adjacency following GF-CF
        rowsum = np.array(self.adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_u = sp.diags(d_inv)
        
        colsum = np.array(self.adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()  
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_i = sp.diags(d_inv)
        
        # Normalized adjacency matrix
        norm_adj = d_mat_u.dot(self.adj_mat).dot(d_mat_i)
        
        # Convert to dense tensor for GPU operations
        self.register_buffer('norm_adj', torch.tensor(norm_adj.toarray(), dtype=torch.float32).to(self.device))
        print("Two-hop setup complete")
    
    def _compute_user_similarity(self):
        """Compute user-user similarity with GF-CF normalization"""
        # Row normalization
        rowsum = np.array(self.adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(self.adj_mat)
        
        # Column normalization
        colsum = np.array(self.adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        
        # User-user similarity
        return norm_adj @ norm_adj.T
    
    def _compute_item_similarity(self):
        """Compute item-item similarity with learnable gamma normalization"""
        if self.learnable_gamma and hasattr(self, 'item_gamma'):
            # Use learnable gamma parameter: D^{-γ} R D^{γ-1}
            gamma = self.item_gamma.item()  # Get current value
            print(f"Using learnable gamma normalization: γ={gamma:.3f}")
            
            # User degree normalization: D_u^{-γ}
            rowsum = np.array(self.adj_mat.sum(axis=1))
            d_inv_u = np.power(rowsum, -gamma).flatten()
            d_inv_u[np.isinf(d_inv_u)] = 0.
            d_mat_u = sp.diags(d_inv_u)
            
            # Item degree normalization: D_i^{γ-1}
            colsum = np.array(self.adj_mat.sum(axis=0))
            d_inv_i = np.power(colsum, gamma - 1.0).flatten()
            d_inv_i[np.isinf(d_inv_i)] = 0.
            d_mat_i = sp.diags(d_inv_i)
            
            # Apply generalized normalization
            norm_adj = d_mat_u.dot(self.adj_mat).dot(d_mat_i)
        else:
            # Standard GF-CF normalization (γ=0.5)
            rowsum = np.array(self.adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_adj = d_mat.dot(self.adj_mat)
            
            colsum = np.array(self.adj_mat.sum(axis=0))
            d_inv = np.power(colsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_adj = norm_adj.dot(d_mat)
        
        # Item-item similarity
        return norm_adj.T @ norm_adj
    
    def _compute_bipartite_similarity(self):
        """Compute bipartite graph similarity"""
        # GF-CF normalization
        rowsum = np.array(self.adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(self.adj_mat)
        
        colsum = np.array(self.adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        
        # Bipartite adjacency matrix
        return sp.bmat([[None, norm_adj], [norm_adj.T, None]], format='csr')
    
    def forward(self, users):
        """Forward pass - generate recommendations"""
        if not isinstance(users, torch.Tensor):
            users = torch.tensor(users, dtype=torch.long).to(self.device)
        
        # Get user profiles
        user_profiles = self.adj_tensor[users]  # (batch_size, n_items)
        
        # Raw-only mode: just return two-hop propagation
        if self.raw_only:
            return user_profiles @ self.norm_adj.T @ self.norm_adj
        
        scores = []
        
        # User view filtering
        if 'u' in self.filter_views and hasattr(self, 'user_eigenvals'):
            # Apply learnable filter to eigenvalues
            filter_response = self.user_filter(self.user_eigenvals)
            
            # User filtering
            batch_user_vecs = self.user_eigenvecs[users]  # (batch, n_eigen)
            user_filtered = batch_user_vecs @ torch.diag(filter_response) @ batch_user_vecs.T @ user_profiles
            scores.append(user_filtered)
        
        # Item view filtering
        if 'i' in self.filter_views and hasattr(self, 'item_eigenvals'):
            filter_response = self.item_filter(self.item_eigenvals)
            
            # Check if we need to recompute item similarity with learnable gamma
            if self.learnable_gamma and hasattr(self, 'item_gamma'):
                item_eigenvecs, item_eigenvals = self._compute_item_eigenvectors_dynamic()
                item_filtered = user_profiles @ item_eigenvecs @ torch.diag(filter_response) @ item_eigenvecs.T
            else:
                # Standard item filtering
                item_filtered = user_profiles @ self.item_eigenvecs @ torch.diag(filter_response) @ self.item_eigenvecs.T
            
            scores.append(item_filtered)
        
        # Bipartite view filtering
        if 'b' in self.filter_views and hasattr(self, 'bipartite_eigenvals'):
            filter_response = self.bipartite_filter(self.bipartite_eigenvals)
            
            # Bipartite filtering
            batch_bipartite_vecs = self.bipartite_eigenvecs[users]  # (batch, n_eigen)
            bipartite_filtered = batch_bipartite_vecs @ torch.diag(filter_response) @ batch_bipartite_vecs.T @ user_profiles
            scores.append(bipartite_filtered)
        
        # Combine scores
        if not scores:
            final_scores = user_profiles
        else:
            final_scores = sum(scores) / len(scores)
        
        # Add two-hop propagation if enabled
        if self.use_two_hop:
            # Two-hop: user_profiles @ norm_adj.T @ norm_adj
            two_hop_scores = user_profiles @ self.norm_adj.T @ self.norm_adj
            
            # Combine with learnable weight
            if self.dataset == 'amazon-book':
                # For Amazon-book, GF-CF only uses two-hop
                final_scores = two_hop_scores
            else:
                # For other datasets, combine spectral and two-hop
                final_scores = final_scores + self.two_hop_weight * two_hop_scores
        
        return final_scores
    
    def getUsersRating(self, batch_users):
        """Get ratings for evaluation (interface compatibility)"""
        self.eval()
        with torch.no_grad():
            scores = self.forward(batch_users)
            return scores.cpu().numpy()
    
    def get_optimizer_groups(self):
        """Get parameter groups for per-view optimization"""
        groups = []
        
        if hasattr(self, 'user_filter'):
            groups.append({
                'params': self.user_filter.parameters(),
                'lr': self.user_lr,
                'weight_decay': self.user_decay,
                'name': 'user_filter'
            })
        
        if hasattr(self, 'item_filter'):
            groups.append({
                'params': self.item_filter.parameters(),
                'lr': self.item_lr,
                'weight_decay': self.item_decay,
                'name': 'item_filter'
            })
        
        if hasattr(self, 'bipartite_filter'):
            groups.append({
                'params': self.bipartite_filter.parameters(),
                'lr': self.bipartite_lr,
                'weight_decay': self.bipartite_decay,
                'name': 'bipartite_filter'
            })
        
        # Add two-hop weight if enabled
        if self.use_two_hop and self.dataset != 'amazon-book':
            groups.append({
                'params': [self.two_hop_weight],
                'lr': 0.01,  # Use a moderate learning rate for the weight
                'weight_decay': 0,
                'name': 'two_hop_weight'
            })
        
        # Add learnable gamma parameter if enabled
        if self.learnable_gamma and hasattr(self, 'item_gamma'):
            groups.append({
                'params': [self.item_gamma],
                'lr': 0.001,  # Small learning rate for gamma parameter
                'weight_decay': 0,
                'name': 'item_gamma'
            })
        
        return groups