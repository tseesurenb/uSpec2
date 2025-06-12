'''
Created on June 12, 2025
Improved Universal Spectral CF with BPR Loss and Advanced Training
Based on DySimGCF's successful training methodology
WITH SIMILARITY MATRIX CACHING FOR PERFORMANCE

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import gc
import os
import pickle
import filters as fl


class ImprovedUniversalSpectralCF(nn.Module):
    """Universal Spectral CF with BPR Loss and Improved Training"""
    
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        # Basic configuration
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.filter_order = self.config.get('filter_order', 6)
        self.filter = self.config.get('filter', 'ui')
        self.dataset = self.config.get('dataset', 'unknown')
        
        # Convert adjacency matrix
        adj_dense = adj_mat.toarray() if sp.issparse(adj_mat) else adj_mat
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32).to(self.device))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Eigenvalue counts (use config values or auto-calculate)
        print(f"DEBUG: Config eigenvalues - u={self.config.get('u_n_eigen')}, i={self.config.get('i_n_eigen')}, b={self.config.get('b_n_eigen')}")
        
        if self.config.get('u_n_eigen') in [None, 0]:
            self.u_n_eigen = min(32, max(16, self.n_users // 50))
        else:
            self.u_n_eigen = self.config.get('u_n_eigen')
            
        if self.config.get('i_n_eigen') in [None, 0]:
            self.i_n_eigen = min(48, max(24, self.n_items // 50))
        else:
            self.i_n_eigen = self.config.get('i_n_eigen')
            
        if self.config.get('b_n_eigen') in [None, 0]:
            self.b_n_eigen = min(40, max(20, (self.n_users + self.n_items) // 80))
        else:
            self.b_n_eigen = self.config.get('b_n_eigen')
        
        print(f"Initializing {self.dataset}: {self.n_users} users, {self.n_items} items")
        print(f"Filter: {self.filter}, Eigenvalues: u={self.u_n_eigen}, i={self.i_n_eigen}, b={self.b_n_eigen}")
        
        # Setup spectral filters
        self.setup_spectral_filters()
        self.setup_combination_weights()
        
        # Add embedding layers for better personalization
        embed_dim = self.config.get('embed_dim', 64)
        self.user_embedding = nn.Embedding(self.n_users, embed_dim)
        self.item_embedding = nn.Embedding(self.n_items, embed_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def setup_spectral_filters(self):
        """Setup spectral filters with simplified designs"""
        start = time.time()
        
        # Use simpler, more effective filters
        if self.filter in ['u', 'ui', 'uib', 'ub']:
            print("Computing user-user similarity...")
            self.user_eigenvals, self.user_eigenvecs = self.compute_user_eigen()
            self.user_filter = self.create_simple_filter()
        
        if self.filter in ['i', 'ui', 'uib']:
            print("Computing item-item similarity...")
            self.item_eigenvals, self.item_eigenvecs = self.compute_item_eigen()
            self.item_filter = self.create_simple_filter()
        
        if self.filter in ['b', 'uib', 'ub']:
            print("Computing bipartite structure...")
            self.bipartite_eigenvals, self.bipartite_eigenvecs = self.compute_bipartite_eigen()
            self.bipartite_filter = self.create_simple_filter()
        
        print(f'Spectral setup completed in {time.time() - start:.2f}s')
    
    def compute_user_eigen(self):
        """Improved user eigendecomposition with similarity matrix caching"""
        # Cache paths for both eigendecomposition and similarity matrix
        eigen_cache_path = f"../cache/{self.dataset}_user_eigen_{self.u_n_eigen}.pkl"
        similarity_cache_path = f"../cache/{self.dataset}_user_similarity.pkl"
        
        # Check if both cached files exist
        if os.path.exists(eigen_cache_path) and os.path.exists(similarity_cache_path):
            print("Loading cached user eigendecomposition and similarity matrix...")
            with open(eigen_cache_path, 'rb') as f:
                eigenvals, eigenvecs = pickle.load(f)
            with open(similarity_cache_path, 'rb') as f:
                self.user_similarity_matrix = pickle.load(f).to(self.device)
            return eigenvals.to(self.device), eigenvecs.to(self.device)
        
        print("Computing user-user similarity matrix...")
        
        # Compute similarity matrix (most expensive operation)
        degree = self.adj_tensor.sum(dim=1) + 1e-8
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        # User similarity: A @ A^T
        user_adj = self.adj_tensor @ self.adj_tensor.t()
        normalized_adj = user_adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
        
        # Cache the similarity matrix
        os.makedirs("../cache", exist_ok=True)
        print("Caching user similarity matrix...")
        with open(similarity_cache_path, 'wb') as f:
            pickle.dump(normalized_adj.cpu(), f)
        self.user_similarity_matrix = normalized_adj
        
        # Compute eigendecomposition
        k = min(self.u_n_eigen, self.n_users - 2)
        eigenvals, eigenvecs = eigsh(sp.csr_matrix(normalized_adj.cpu().numpy()), k=k, which='LM')
        eigenvals = np.maximum(eigenvals, 0.0)
        
        eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
        eigenvecs_tensor = torch.tensor(eigenvecs, dtype=torch.float32)
        
        # Cache eigendecomposition
        print("Caching user eigendecomposition...")
        with open(eigen_cache_path, 'wb') as f:
            pickle.dump((eigenvals_tensor, eigenvecs_tensor), f)
        
        return eigenvals_tensor.to(self.device), eigenvecs_tensor.to(self.device)
    
    def compute_item_eigen(self):
        """Improved item eigendecomposition with similarity matrix caching"""
        eigen_cache_path = f"../cache/{self.dataset}_item_eigen_{self.i_n_eigen}.pkl"
        similarity_cache_path = f"../cache/{self.dataset}_item_similarity.pkl"
        
        if os.path.exists(eigen_cache_path) and os.path.exists(similarity_cache_path):
            print("Loading cached item eigendecomposition and similarity matrix...")
            with open(eigen_cache_path, 'rb') as f:
                eigenvals, eigenvecs = pickle.load(f)
            with open(similarity_cache_path, 'rb') as f:
                self.item_similarity_matrix = pickle.load(f).to(self.device)
            return eigenvals.to(self.device), eigenvecs.to(self.device)
        
        print("Computing item-item similarity matrix...")
        
        # Compute item similarity
        adj_t = self.adj_tensor.t()
        degree = adj_t.sum(dim=1) + 1e-8
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        item_adj = adj_t @ adj_t.t()
        normalized_adj = item_adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
        
        # Cache similarity matrix
        os.makedirs("../cache", exist_ok=True)
        print("Caching item similarity matrix...")
        with open(similarity_cache_path, 'wb') as f:
            pickle.dump(normalized_adj.cpu(), f)
        self.item_similarity_matrix = normalized_adj
        
        # Eigendecomposition
        k = min(self.i_n_eigen, self.n_items - 2)
        eigenvals, eigenvecs = eigsh(sp.csr_matrix(normalized_adj.cpu().numpy()), k=k, which='LM')
        eigenvals = np.maximum(eigenvals, 0.0)
        
        eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
        eigenvecs_tensor = torch.tensor(eigenvecs, dtype=torch.float32)
        
        print("Caching item eigendecomposition...")
        with open(eigen_cache_path, 'wb') as f:
            pickle.dump((eigenvals_tensor, eigenvecs_tensor), f)
        
        return eigenvals_tensor.to(self.device), eigenvecs_tensor.to(self.device)
    
    def compute_bipartite_eigen(self):
        """Improved bipartite eigendecomposition with similarity matrix caching"""
        eigen_cache_path = f"../cache/{self.dataset}_bipartite_eigen_{self.b_n_eigen}.pkl"
        similarity_cache_path = f"../cache/{self.dataset}_bipartite_similarity.pkl"
        
        if os.path.exists(eigen_cache_path) and os.path.exists(similarity_cache_path):
            print("Loading cached bipartite eigendecomposition and similarity matrix...")
            with open(eigen_cache_path, 'rb') as f:
                eigenvals, eigenvecs = pickle.load(f)
            with open(similarity_cache_path, 'rb') as f:
                self.bipartite_similarity_matrix = pickle.load(f).to(self.device)
            return eigenvals.to(self.device), eigenvecs.to(self.device)
        
        print("Computing bipartite similarity matrix...")
        
        # Create bipartite adjacency
        n_total = self.n_users + self.n_items
        bipartite_adj = torch.zeros(n_total, n_total, device=self.device)
        bipartite_adj[:self.n_users, self.n_users:] = self.adj_tensor
        bipartite_adj[self.n_users:, :self.n_users] = self.adj_tensor.t()
        
        # Normalize
        degree = bipartite_adj.sum(dim=1) + 1e-8
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        normalized_adj = bipartite_adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
        
        # Cache similarity matrix
        os.makedirs("../cache", exist_ok=True)
        print("Caching bipartite similarity matrix...")
        with open(similarity_cache_path, 'wb') as f:
            pickle.dump(normalized_adj.cpu(), f)
        self.bipartite_similarity_matrix = normalized_adj
        
        # Eigendecomposition
        k = min(self.b_n_eigen, n_total - 2)
        eigenvals, eigenvecs = eigsh(sp.csr_matrix(normalized_adj.cpu().numpy()), k=k, which='LM')
        eigenvals = np.maximum(eigenvals, 0.0)
        
        eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
        eigenvecs_tensor = torch.tensor(eigenvecs, dtype=torch.float32)
        
        print("Caching bipartite eigendecomposition...")
        with open(eigen_cache_path, 'wb') as f:
            pickle.dump((eigenvals_tensor, eigenvecs_tensor), f)
        
        return eigenvals_tensor.to(self.device), eigenvecs_tensor.to(self.device)
    
    def create_simple_filter(self):
        """Create simple but effective spectral filter"""
        return SimpleSpectralFilter(self.filter_order)
    
    def setup_combination_weights(self):
        """Setup learnable combination weights"""
        if self.filter == 'u':
            init_weights = torch.tensor([0.6, 0.4])  # [spectral, embedding]
        elif self.filter == 'i':
            init_weights = torch.tensor([0.6, 0.4])
        elif self.filter == 'b':
            init_weights = torch.tensor([0.6, 0.4])
        elif self.filter == 'ui':
            init_weights = torch.tensor([0.4, 0.3, 0.3])  # [user_spectral, item_spectral, embedding]
        elif self.filter == 'ub':
            init_weights = torch.tensor([0.4, 0.3, 0.3])
        elif self.filter == 'uib':
            init_weights = torch.tensor([0.3, 0.25, 0.25, 0.2])
        else:
            init_weights = torch.tensor([0.6, 0.4])
        
        self.combination_weights = nn.Parameter(init_weights.to(self.device))
    
    def forward(self, users):
        """Forward pass using cached similarity matrices for better performance"""
        if users.device != self.adj_tensor.device:
            users = users.to(self.adj_tensor.device)
        
        batch_size = users.shape[0]
        user_profiles = self.adj_tensor[users]
        
        scores = []
        
        # Embedding-based collaborative filtering
        user_embeds = self.user_embedding(users)
        item_embeds = self.item_embedding.weight
        embedding_scores = user_embeds @ item_embeds.t()
        scores.append(embedding_scores)
        
        # User-based spectral filtering (using cached similarity matrix)
        if self.filter in ['u', 'ui', 'uib', 'ub'] and hasattr(self, 'user_filter'):
            user_responses = self.user_filter(self.user_eigenvals)
            avg_user_response = user_responses.mean() if user_responses.dim() > 0 else user_responses
            
            # Use precomputed similarity matrix instead of recomputing
            user_filtered = self.user_similarity_matrix[users] @ self.adj_tensor
            scores.append(user_filtered * avg_user_response)
        
        # Item-based spectral filtering (using cached similarity matrix)
        if self.filter in ['i', 'ui', 'uib'] and hasattr(self, 'item_filter'):
            item_responses = self.item_filter(self.item_eigenvals)
            avg_item_response = item_responses.mean() if item_responses.dim() > 0 else item_responses
            
            # Use precomputed similarity matrix
            item_filtered = user_profiles @ self.item_similarity_matrix
            scores.append(item_filtered * avg_item_response)
        
        # Bipartite spectral filtering (using cached similarity matrix)
        if self.filter in ['b', 'uib', 'ub'] and hasattr(self, 'bipartite_filter'):
            bipartite_responses = self.bipartite_filter(self.bipartite_eigenvals)
            avg_bipartite_response = bipartite_responses.mean() if bipartite_responses.dim() > 0 else bipartite_responses
            
            # Use precomputed bipartite similarity matrix
            item_part = self.bipartite_similarity_matrix[self.n_users:, self.n_users:]
            bipartite_filtered = user_profiles @ item_part
            scores.append(bipartite_filtered * avg_bipartite_response)
        
        # Combine predictions with learnable weights
        weights = torch.softmax(self.combination_weights, dim=0)
        predicted = sum(w * score for w, score in zip(weights, scores))
        
        return predicted
    
    def getUsersRating(self, batch_users):
        """Evaluation interface"""
        self.eval()
        with torch.no_grad():
            if isinstance(batch_users, np.ndarray):
                batch_users = torch.LongTensor(batch_users)
            
            if batch_users.device != self.device:
                batch_users = batch_users.to(self.device)
            
            result = self.forward(batch_users).cpu().numpy()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result


class SimpleSpectralFilter(nn.Module):
    """Simplified spectral filter for better performance"""
    
    def __init__(self, filter_order=6):
        super().__init__()
        self.filter_order = filter_order
        
        # Simple learnable low-pass filter
        self.cutoff = nn.Parameter(torch.tensor(0.5))
        self.sharpness = nn.Parameter(torch.tensor(10.0))
        
    def forward(self, eigenvalues):
        device = eigenvalues.device
        cutoff = torch.sigmoid(self.cutoff).to(device)
        sharpness = torch.abs(self.sharpness).to(device) + 1.0
        
        # Normalize eigenvalues
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval
        
        # Simple sigmoid-based low-pass filter
        response = torch.sigmoid(sharpness * (cutoff - norm_eigenvals)) + 0.1
        
        return response


# Cache management utilities
def clear_similarity_cache(dataset):
    """Clear cached similarity matrices for a dataset"""
    cache_files = [
        f"../cache/{dataset}_user_similarity.pkl",
        f"../cache/{dataset}_item_similarity.pkl", 
        f"../cache/{dataset}_bipartite_similarity.pkl"
    ]
    
    removed_count = 0
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Removed {cache_file}")
            removed_count += 1
    
    if removed_count == 0:
        print(f"No similarity cache files found for dataset: {dataset}")
    else:
        print(f"Cleared {removed_count} similarity cache files for {dataset}")


def get_cache_info(dataset):
    """Get information about cached files"""
    cache_files = [
        f"../cache/{dataset}_user_similarity.pkl",
        f"../cache/{dataset}_item_similarity.pkl",
        f"../cache/{dataset}_bipartite_similarity.pkl"
    ]
    
    print(f"\n📁 Cache Status for {dataset}:")
    print("=" * 50)
    
    total_size = 0
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            total_size += size_mb
            print(f"✅ {os.path.basename(cache_file)}: {size_mb:.1f} MB")
        else:
            print(f"❌ {os.path.basename(cache_file)}: Not cached")
    
    print(f"\n💾 Total similarity cache size: {total_size:.1f} MB")
    
    # Also check eigenvalue caches
    eigen_pattern = f"../cache/{dataset}_*_eigen_*.pkl"
    import glob
    eigen_files = glob.glob(eigen_pattern)
    if eigen_files:
        print(f"\n🔢 Found {len(eigen_files)} eigenvalue cache files")
        for eigen_file in eigen_files:
            size_mb = os.path.getsize(eigen_file) / (1024 * 1024)
            print(f"   {os.path.basename(eigen_file)}: {size_mb:.2f} MB")


# Simple configuration functions
def get_fast_config():
    return {
        'filter_order': 4,
        'embed_dim': 32,
        'epochs': 30
    }

def get_standard_config():
    return {
        'filter_order': 6,
        'embed_dim': 64,
        'epochs': 50
    }


# Example usage for cache management
if __name__ == "__main__":
    # Example: Check cache status
    # get_cache_info('gowalla')
    
    # Example: Clear similarity caches
    # clear_similarity_cache('gowalla')
    pass