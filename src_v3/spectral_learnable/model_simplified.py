'''
Simplified Spectral Model - Similar to static model but still trainable
Uses GF-CF normalization and removes complex features
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import filters as fl
import os
import pickle
import hashlib


class SimplifiedSpectralCF(nn.Module):
    """Simplified Spectral CF - like static model but trainable"""
    
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        # Basic configuration
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.filter = self.config.get('filter', 'ui')  # fallback to 'filter' for compatibility
        self.dataset = self.config.get('dataset', 'unknown')
        
        # Convert adjacency matrix
        adj_dense = adj_mat.toarray() if sp.issparse(adj_mat) else adj_mat
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32).to(self.device))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Store original adjacency matrix for caching (ensure CSR format)
        if sp.issparse(adj_mat):
            self.adj_mat = adj_mat.tocsr()
        else:
            self.adj_mat = sp.csr_matrix(adj_mat)
        
        # Setup cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), '../cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Eigenvalue counts
        self.u_n_eigen = self.config.get('u_n_eigen', 8)
        self.i_n_eigen = self.config.get('i_n_eigen', 40)
        self.b_n_eigen = self.config.get('b_n_eigen', 60)
        
        # Map short names to full filter names (like spectral_clean)
        filter_mapping = {
            'orig': 'original', 'cheby': 'chebyshev', 'jacobi': 'jacobi', 
            'legendre': 'legendre', 'laguerre': 'laguerre', 'hermite': 'hermite',
            'bernstein': 'bernstein', 'multi': 'multiscale', 'band': 'bandstop', 
            'ensemble': 'ensemble', 'golden': 'golden', 'harmonic': 'harmonic',
            'spectral_basis': 'spectral_basis', 'enhanced_basis': 'enhanced_basis'
        }
        
        # Map short names to full initialization names  
        init_mapping = {
            'smooth': 'smooth', 'sharp': 'sharp', 'bandpass': 'bandpass',
            'golden': 'golden_036', 'butter': 'butterworth', 'gauss': 'gaussian',
            'stop': 'band_stop', 'notch': 'notch'
        }
        
        # Get filter designs from config and map to full names
        user_filter_short = self.config.get('user_filter_design', 'orig')
        item_filter_short = self.config.get('item_filter_design', 'orig')
        bipartite_filter_short = self.config.get('bipartite_filter_design', 'orig')
        
        self.user_filter_design = filter_mapping.get(user_filter_short, user_filter_short)
        self.item_filter_design = filter_mapping.get(item_filter_short, item_filter_short)
        self.bipartite_filter_design = filter_mapping.get(bipartite_filter_short, bipartite_filter_short)
        
        # Get filter initializations from config and map to full names
        user_init_short = self.config.get('user_init_filter', 'smooth')
        item_init_short = self.config.get('item_init_filter', 'sharp')
        bipartite_init_short = self.config.get('bipartite_init_filter', 'smooth')
        
        self.user_init_filter = init_mapping.get(user_init_short, user_init_short)
        self.item_init_filter = init_mapping.get(item_init_short, item_init_short)
        self.bipartite_init_filter = init_mapping.get(bipartite_init_short, bipartite_init_short)
        
        print(f"Simplified {self.dataset}: {self.n_users} users, {self.n_items} items")
        print(f"Filter: {self.filter}, Eigenvalues: u={self.u_n_eigen}, i={self.i_n_eigen}, b={self.b_n_eigen}")
        
        
        # Setup filters and eigendecompositions
        self.setup_spectral_filters()
    
    def get_cache_key(self):
        """Generate cache key based on adjacency matrix and normalization method"""
        # Create hash of adjacency matrix content
        adj_hash = hashlib.md5(self.adj_mat.data.tobytes() + 
                              self.adj_mat.indices.tobytes() + 
                              self.adj_mat.indptr.tobytes()).hexdigest()
        
        # Cache key includes matrix hash and normalization type (always GF-CF for this model)
        cache_key = f"gfcf_{self.n_users}_{self.n_items}_{adj_hash[:16]}"
        return cache_key
    
    def load_cached_similarities(self):
        """Load cached similarity matrices if they exist"""
        cache_key = self.get_cache_key()
        cache_file = os.path.join(self.cache_dir, f"similarities_{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                print(f"Loading cached similarity matrices from {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data['user_sim'], cached_data['item_sim'], cached_data['bipartite_sim']
            except Exception as e:
                print(f"Failed to load cache: {e}")
                return None, None, None
        return None, None, None
    
    def save_cached_similarities(self, user_sim, item_sim, bipartite_sim):
        """Save similarity matrices to cache"""
        cache_key = self.get_cache_key()
        cache_file = os.path.join(self.cache_dir, f"similarities_{cache_key}.pkl")
        
        try:
            cached_data = {
                'user_sim': user_sim,
                'item_sim': item_sim, 
                'bipartite_sim': bipartite_sim,
                'n_users': self.n_users,
                'n_items': self.n_items
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"Saved similarity matrices to cache: {cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def setup_spectral_filters(self):
        """Compute eigendecompositions and setup filters"""
        start = time.time()
        
        # Try to load cached similarity matrices
        cached_user_sim, cached_item_sim, cached_bipartite_sim = self.load_cached_similarities()
        
        if cached_user_sim is not None:
            print("Using cached similarity matrices!")
            user_sim, item_sim, bipartite_sim = cached_user_sim, cached_item_sim, cached_bipartite_sim
        else:
            print("Computing similarity matrices (will be cached for future use)...")
            # Compute similarity matrices
            user_sim = self.compute_user_similarity()
            item_sim = self.compute_item_similarity() 
            bipartite_sim = self.compute_bipartite_similarity()
            
            # Save to cache
            self.save_cached_similarities(user_sim, item_sim, bipartite_sim)
        
        # Setup view-specific filters using cached/computed similarities  
        print("Computing eigendecompositions...")
        
        if 'u' in self.filter:
            self.user_eigenvals, self.user_eigenvecs = self.compute_eigen_from_similarity(user_sim, self.u_n_eigen)
        
        if 'i' in self.filter:
            self.item_eigenvals, self.item_eigenvecs = self.compute_eigen_from_similarity(item_sim, self.i_n_eigen)
        
        if 'b' in self.filter:
            self.bipartite_eigenvals, self.bipartite_eigenvecs = self.compute_eigen_from_similarity(bipartite_sim, self.b_n_eigen)
        
        print(f'Training completed in {time.time() - start:.2f}s')
        
        # Debug: Print first few eigenvalues to compare with static model
        if hasattr(self, 'user_eigenvals'):
            print(f"LEARNABLE user eigenvals[0:5]: {self.user_eigenvals[:5]}")
        if hasattr(self, 'item_eigenvals'):
            print(f"LEARNABLE item eigenvals[0:5]: {self.item_eigenvals[:5]}")
        if hasattr(self, 'bipartite_eigenvals'):
            print(f"LEARNABLE bipartite eigenvals[0:5]: {self.bipartite_eigenvals[:5]}")
    
    def compute_user_similarity(self):
        """Compute user similarity matrix with GF-CF normalization (EXACTLY like static model)"""
        # Use exactly the same approach as static model
        adj_mat = self.adj_mat
        
        # Row normalization (user normalization) - exactly like static
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        
        # Column normalization - exactly like static
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        
        # User-user similarity: UU^T (exactly like static model - NO Laplacian!)
        user_sim = norm_adj @ norm_adj.T
        return user_sim
    
    def compute_item_similarity(self):
        """Compute item similarity matrix with GF-CF normalization (EXACTLY like static model)"""
        # Same normalization as static model
        adj_mat = self.adj_mat
        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        
        # Item-item similarity: U^TU (exactly like static model - NO Laplacian!)
        item_sim = norm_adj.T @ norm_adj
        return item_sim
    
    def compute_bipartite_similarity(self):
        """Compute bipartite similarity matrix with GF-CF normalization (EXACTLY like static model)"""
        # Create bipartite adjacency matrix exactly like static model
        adj_mat = self.adj_mat
        n_users, n_items = adj_mat.shape
        
        # Build bipartite matrix [0, U; U^T, 0] with GF-CF normalization (exactly like static)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        
        # Create bipartite matrix exactly like static model
        bipartite = sp.bmat([[None, norm_adj], [norm_adj.T, None]], format='csr')
        return bipartite
    
    def compute_eigen_from_similarity(self, similarity_matrix, k):
        """Compute eigendecomposition from a similarity matrix (EXACTLY like static model)"""
        eigenvals, eigenvecs = eigsh(similarity_matrix, k=min(k, similarity_matrix.shape[0]-1), which='LM')
        eigenvals = np.real(eigenvals)
        eigenvecs = np.real(eigenvecs)
        return torch.tensor(eigenvals, dtype=torch.float32).to(self.device), \
               torch.tensor(eigenvecs, dtype=torch.float32).to(self.device)
    
    def create_spectral_filter(self, eigenvals, filter_design, init_pattern):
        """Create spectral filter using filters.py factory (like static model)"""
        # Create filter object using the filters.py factory
        filter_obj = fl.create_filter(filter_design, filter_order=6, init_filter_name=init_pattern)
        
        # Apply filter to eigenvalues
        if hasattr(filter_obj, 'forward'):
            # Filter object has forward method
            filter_values = filter_obj.forward(eigenvals.unsqueeze(0)).squeeze(0)
        elif hasattr(filter_obj, '__call__'):
            # Filter object is callable
            filter_values = filter_obj(eigenvals)
        else:
            # Fallback to simple polynomial
            filter_values = torch.pow(1.0 - eigenvals, 6)
            
        return filter_values
    
    def forward(self, users):
        """Forward pass - similar to getUsersRating in static model"""
        if users.device != self.adj_tensor.device:
            users = users.to(self.adj_tensor.device)
        
        batch_size = users.shape[0]
        user_profiles = self.adj_tensor[users]
        
        scores = []
        
        # User view filtering (only if active in filter)
        if 'u' in self.filter and hasattr(self, 'user_eigenvals'):
            user_filter = self.create_spectral_filter(self.user_eigenvals, self.user_filter_design, self.user_init_filter)
            batch_user_vecs = self.user_eigenvecs[users, :]  # (batch_size, u_n_eigen)
            user_filtered = batch_user_vecs @ torch.diag(user_filter) @ batch_user_vecs.t() @ user_profiles
            scores.append(user_filtered)
        
        # Item view filtering (only if active in filter)
        if 'i' in self.filter and hasattr(self, 'item_eigenvals'):
            item_filter = self.create_spectral_filter(self.item_eigenvals, self.item_filter_design, self.item_init_filter)
            item_filtered = user_profiles @ self.item_eigenvecs @ torch.diag(item_filter) @ self.item_eigenvecs.t()
            scores.append(item_filtered)
        
        # Bipartite view filtering (only if active in filter)
        if 'b' in self.filter and hasattr(self, 'bipartite_eigenvals'):
            bipartite_user_vecs = self.bipartite_eigenvecs[users, :]  # (batch_size, b_n_eigen)
            bipartite_filter = self.create_spectral_filter(self.bipartite_eigenvals, self.bipartite_filter_design, self.bipartite_init_filter)
            bipartite_filtered = bipartite_user_vecs @ torch.diag(bipartite_filter) @ bipartite_user_vecs.t() @ user_profiles
            scores.append(bipartite_filtered)
        
        # If no filters are active, just return user profiles
        if not scores:
            return user_profiles
        
        # Combine predictions with fixed equal weights (like static model)
        predicted = sum(scores) / len(scores)
        
        return predicted
    
    def getUsersRating(self, batch_users, ds_name=None):
        """Evaluation interface - matches static model signature"""
        self.eval()
        with torch.no_grad():
            # Convert to numpy array like the static model
            batch_users = np.array(batch_users)
            
            # User profiles (convert to scipy sparse like static model)
            user_profiles = self.adj_tensor[batch_users]  # (batch_size, n_items)
            
            
            scores = []
            
            # User view filtering (only if active in filter)
            if 'u' in self.filter and hasattr(self, 'user_eigenvals'):
                user_filter = self.create_spectral_filter(self.user_eigenvals, self.user_filter_design, self.user_init_filter)
                batch_user_vecs = self.user_eigenvecs[batch_users, :]  # (batch_size, u_n_eigen)
                user_filtered = batch_user_vecs @ torch.diag(user_filter) @ batch_user_vecs.t() @ user_profiles
                scores.append(user_filtered)
            
            # Item view filtering (only if active in filter)
            if 'i' in self.filter and hasattr(self, 'item_eigenvals'):
                item_filter = self.create_spectral_filter(self.item_eigenvals, self.item_filter_design, self.item_init_filter)
                item_filtered = user_profiles @ self.item_eigenvecs @ torch.diag(item_filter) @ self.item_eigenvecs.t()
                scores.append(item_filtered)
            
            # Bipartite view filtering (only if active in filter)
            if 'b' in self.filter and hasattr(self, 'bipartite_eigenvals'):
                bipartite_user_vecs = self.bipartite_eigenvecs[batch_users, :]  # (batch_size, b_n_eigen)
                bipartite_filter = self.create_spectral_filter(self.bipartite_eigenvals, self.bipartite_filter_design, self.bipartite_init_filter)
                bipartite_filtered = bipartite_user_vecs @ torch.diag(bipartite_filter) @ bipartite_user_vecs.t() @ user_profiles
                scores.append(bipartite_filtered)
            
            # If no filters are active, just return user profiles
            if not scores:
                return user_profiles.cpu().numpy()
            
            # Combine predictions with fixed equal weights (like static model)
            final_scores = sum(scores) / len(scores)
            
            return final_scores.cpu().numpy()


# Create alias for compatibility
UserSpecificUniversalSpectralCF = SimplifiedSpectralCF