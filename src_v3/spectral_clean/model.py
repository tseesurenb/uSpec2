"""
Clean Spectral Model - Borrowing normalization ideas from GF-CF
"""
import time
import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import eigsh
import torch
import filters as fl
import os
import pickle
import hashlib


class SpectralCF(object):
    def __init__(self, adj_mat, config=None):
        self.adj_mat = adj_mat
        self.n_users, self.n_items = adj_mat.shape
        
        # Setup cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), '../cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Get input matrix filter and eigenvalue counts from config or use defaults
        self.filter = config.get('in_mat', 'uib') if config else 'uib'
        self.u_n_eigen = config.get('u_n_eigen', 8) if config else 8
        self.i_n_eigen = config.get('i_n_eigen', 40) if config else 40
        self.b_n_eigen = config.get('b_n_eigen', 60) if config else 60
        
        # Map short names to full filter names
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
        
        # Get filter designs from config or use defaults
        user_filter_short = config.get('user_filter_design', 'orig') if config else 'orig'
        item_filter_short = config.get('item_filter_design', 'orig') if config else 'orig'
        bipartite_filter_short = config.get('bipartite_filter_design', 'orig') if config else 'orig'
        
        self.user_filter_design = filter_mapping.get(user_filter_short, 'original')
        self.item_filter_design = filter_mapping.get(item_filter_short, 'original')
        self.bipartite_filter_design = filter_mapping.get(bipartite_filter_short, 'original')
        
        # Get filter initializations from config or use defaults
        user_init_short = config.get('user_init_filter', 'smooth') if config else 'smooth'
        item_init_short = config.get('item_init_filter', 'sharp') if config else 'sharp'
        bipartite_init_short = config.get('bipartite_init_filter', 'smooth') if config else 'smooth'
        
        self.user_init_filter = init_mapping.get(user_init_short, 'smooth')
        self.item_init_filter = init_mapping.get(item_init_short, 'sharp')
        self.bipartite_init_filter = init_mapping.get(bipartite_init_short, 'smooth')
        
        print(f"Spectral CF: {self.n_users} users, {self.n_items} items")
        print(f"Input matrix: {self.filter}")
        print(f"Eigenvalues: u={self.u_n_eigen}, i={self.i_n_eigen}, b={self.b_n_eigen}")
        print(f"Filters: user={self.user_filter_design}, item={self.item_filter_design}, bipartite={self.bipartite_filter_design}")
        
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
    
        
        
    def train(self):
        """Compute eigendecompositions for three views"""
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
        
        # Compute eigendecompositions from cached/computed similarities (only for active views)
        print("Computing eigendecompositions...")
        
        if 'u' in self.filter:
            self.user_eigenvals, self.user_eigenvecs = self.compute_eigen_from_similarity(user_sim, self.u_n_eigen)
            print(f"STATIC user eigenvals[0:5]: {self.user_eigenvals[:5]}")
        
        if 'i' in self.filter:
            self.item_eigenvals, self.item_eigenvecs = self.compute_eigen_from_similarity(item_sim, self.i_n_eigen)
            print(f"STATIC item eigenvals[0:5]: {self.item_eigenvals[:5]}")
        
        if 'b' in self.filter:
            self.bipartite_eigenvals, self.bipartite_eigenvecs = self.compute_eigen_from_similarity(bipartite_sim, self.b_n_eigen)
            print(f"STATIC bipartite eigenvals[0:5]: {self.bipartite_eigenvals[:5]}")
        
        print(f'Spectral training completed in {time.time() - start:.2f}s')
        
    def compute_user_similarity(self):
        """Compute user similarity matrix with GF-CF normalization"""
        adj_mat = self.adj_mat
        
        # Row normalization (user normalization)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        
        # Column normalization  
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        
        # User-user similarity: UU^T
        user_sim = norm_adj @ norm_adj.T
        return user_sim
    
    def compute_item_similarity(self):
        """Compute item similarity matrix with GF-CF normalization"""
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
        
        # Item-item similarity: U^TU
        item_sim = norm_adj.T @ norm_adj
        return item_sim
    
    def compute_bipartite_similarity(self):
        """Compute bipartite similarity matrix with GF-CF normalization"""
        adj_mat = self.adj_mat
        n_users, n_items = adj_mat.shape
        
        # Build bipartite matrix [0, U; U^T, 0] with GF-CF normalization
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
        
        # Create bipartite matrix
        bipartite = sp.bmat([[None, norm_adj], [norm_adj.T, None]], format='csr')
        return bipartite
    
    def compute_eigen_from_similarity(self, similarity_matrix, k):
        """Compute eigendecomposition from a similarity matrix"""
        eigenvals, eigenvecs = eigsh(similarity_matrix, k=min(k, similarity_matrix.shape[0]-1), which='LM')
        eigenvals = np.real(eigenvals)
        eigenvecs = np.real(eigenvecs)
        return torch.tensor(eigenvals, dtype=torch.float32), torch.tensor(eigenvecs, dtype=torch.float32)
    
    def getUsersRating(self, batch_users, ds_name):
        """Generate recommendations using spectral filtering based on in_mat parameter"""
        batch_users = np.array(batch_users)
        
        # User profiles  
        user_profiles = self.adj_mat[batch_users].toarray()  # (batch_size, n_items)
        user_profiles_tensor = torch.tensor(user_profiles, dtype=torch.float32)
        
        scores = []
        
        # User view filtering (only if 'u' in filter)
        if 'u' in self.filter:
            user_filter = self.create_spectral_filter(self.user_eigenvals, self.user_filter_design, self.user_init_filter)
            batch_user_vecs = self.user_eigenvecs[batch_users, :]  # (batch_size, u_n_eigen)
            user_filtered = batch_user_vecs @ torch.diag(user_filter) @ batch_user_vecs.T @ user_profiles_tensor
            scores.append(user_filtered)
        
        # Item view filtering (only if 'i' in filter)
        if 'i' in self.filter:
            item_filter = self.create_spectral_filter(self.item_eigenvals, self.item_filter_design, self.item_init_filter)
            item_filtered = user_profiles_tensor @ self.item_eigenvecs @ torch.diag(item_filter) @ self.item_eigenvecs.T
            scores.append(item_filtered)
        
        # Bipartite view filtering (only if 'b' in filter)
        if 'b' in self.filter:
            bipartite_user_vecs = self.bipartite_eigenvecs[batch_users, :]  # (batch_size, b_n_eigen)
            bipartite_filter = self.create_spectral_filter(self.bipartite_eigenvals, self.bipartite_filter_design, self.bipartite_init_filter)
            bipartite_filtered = bipartite_user_vecs @ torch.diag(bipartite_filter) @ bipartite_user_vecs.T @ user_profiles_tensor
            scores.append(bipartite_filtered)
        
        # If no filters are active, just return user profiles
        if not scores:
            return user_profiles
        
        # Combine active views with equal weights
        final_scores = sum(scores) / len(scores)
        
        return final_scores.detach().numpy()
    
    def create_spectral_filter(self, eigenvals, filter_design, init_pattern):
        """Create spectral filter using proven designs and initialization patterns"""
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