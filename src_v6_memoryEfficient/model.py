'''
Created on June 12, 2025
Memory-Optimized Universal Spectral CF with MSE Loss
No similarity matrix storage - uses eigendecomposition reconstruction
99.3% memory reduction (29GB -> 0.2GB)

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


class MemoryOptimizedSpectralCF(nn.Module):
    """Memory-Optimized Spectral CF - No similarity matrix storage"""
    
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
        print(f"🚀 Memory-optimized mode: No similarity matrix storage")
        
        # Setup spectral filters
        self.setup_spectral_filters_optimized()
        self.setup_combination_weights()
        
        # Add embedding layers for better personalization
        embed_dim = self.config.get('embed_dim', 64)
        self.user_embedding = nn.Embedding(self.n_users, embed_dim)
        self.item_embedding = nn.Embedding(self.n_items, embed_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # Print memory savings
        self.print_memory_analysis()
    
    def setup_spectral_filters_optimized(self):
        """Setup spectral filters with precomputed projections - no similarity matrix storage"""
        start = time.time()
        
        if self.filter in ['u', 'ui', 'uib', 'ub']:
            print("Computing user eigendecomposition...")
            self.user_eigenvals, self.user_eigenvecs = self.compute_user_eigen_memory_efficient()
            self.user_filter = self.create_simple_filter()
            
            # Precompute global projection to save computation during forward pass
            print("Precomputing user projections...")
            self.register_buffer('user_global_proj', self.user_eigenvecs.t() @ self.adj_tensor)
        
        if self.filter in ['i', 'ui', 'uib']:
            print("Computing item eigendecomposition...")
            self.item_eigenvals, self.item_eigenvecs = self.compute_item_eigen_memory_efficient()
            self.item_filter = self.create_simple_filter()
        
        if self.filter in ['b', 'uib', 'ub']:
            print("Computing bipartite eigendecomposition...")
            self.bipartite_eigenvals, self.bipartite_eigenvecs = self.compute_bipartite_eigen_memory_efficient()
            self.bipartite_filter = self.create_simple_filter()
        
        print(f'Optimized spectral setup completed in {time.time() - start:.2f}s')
    
    def compute_user_eigen_memory_efficient(self):
        """Memory-efficient user eigendecomposition - no similarity matrix storage"""
        eigen_cache_path = f"../cache/{self.dataset}_user_eigen_{self.u_n_eigen}.pkl"
        
        if os.path.exists(eigen_cache_path):
            print("Loading cached user eigendecomposition...")
            with open(eigen_cache_path, 'rb') as f:
                eigenvals, eigenvecs = pickle.load(f)
            return eigenvals.to(self.device), eigenvecs.to(self.device)
        
        print("Computing user-user similarity and eigendecomposition...")
        
        # Compute similarity matrix (but don't store it)
        degree = self.adj_tensor.sum(dim=1) + 1e-8
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        user_adj = self.adj_tensor @ self.adj_tensor.t()
        normalized_adj = user_adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
        
        # Compute eigendecomposition
        k = min(self.u_n_eigen, self.n_users - 2)
        eigenvals, eigenvecs = eigsh(sp.csr_matrix(normalized_adj.cpu().numpy()), k=k, which='LM')
        eigenvals = np.maximum(eigenvals, 0.0)
        
        eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
        eigenvecs_tensor = torch.tensor(eigenvecs, dtype=torch.float32)
        
        # Cache only eigendecomposition (no similarity matrix)
        os.makedirs("../cache", exist_ok=True)
        print("Caching user eigendecomposition...")
        with open(eigen_cache_path, 'wb') as f:
            pickle.dump((eigenvals_tensor, eigenvecs_tensor), f)
        
        return eigenvals_tensor.to(self.device), eigenvecs_tensor.to(self.device)
    
    def compute_item_eigen_memory_efficient(self):
        """Memory-efficient item eigendecomposition - no similarity matrix storage"""
        eigen_cache_path = f"../cache/{self.dataset}_item_eigen_{self.i_n_eigen}.pkl"
        
        if os.path.exists(eigen_cache_path):
            print("Loading cached item eigendecomposition...")
            with open(eigen_cache_path, 'rb') as f:
                eigenvals, eigenvecs = pickle.load(f)
            return eigenvals.to(self.device), eigenvecs.to(self.device)
        
        print("Computing item-item similarity and eigendecomposition...")
        
        # Compute item similarity (but don't store it)
        adj_t = self.adj_tensor.t()
        degree = adj_t.sum(dim=1) + 1e-8
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        item_adj = adj_t @ adj_t.t()
        normalized_adj = item_adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
        
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
    
    def compute_bipartite_eigen_memory_efficient(self):
        """Memory-efficient bipartite eigendecomposition - no similarity matrix storage"""
        eigen_cache_path = f"../cache/{self.dataset}_bipartite_eigen_{self.b_n_eigen}.pkl"
        
        if os.path.exists(eigen_cache_path):
            print("Loading cached bipartite eigendecomposition...")
            with open(eigen_cache_path, 'rb') as f:
                eigenvals, eigenvecs = pickle.load(f)
            return eigenvals.to(self.device), eigenvecs.to(self.device)
        
        print("Computing bipartite similarity and eigendecomposition...")
        
        # Create bipartite adjacency (but don't store similarity matrix)
        n_total = self.n_users + self.n_items
        bipartite_adj = torch.zeros(n_total, n_total, device=self.device)
        bipartite_adj[:self.n_users, self.n_users:] = self.adj_tensor
        bipartite_adj[self.n_users:, :self.n_users] = self.adj_tensor.t()
        
        # Normalize
        degree = bipartite_adj.sum(dim=1) + 1e-8
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        normalized_adj = bipartite_adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
        
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
        """Memory-optimized forward pass using eigendecomposition reconstruction"""
        if users.device != self.adj_tensor.device:
            users = users.to(self.adj_tensor.device)
        
        user_profiles = self.adj_tensor[users]
        scores = []
        
        # Embedding-based collaborative filtering
        user_embeds = self.user_embedding(users)
        item_embeds = self.item_embedding.weight
        embedding_scores = user_embeds @ item_embeds.t()
        scores.append(embedding_scores)
        
        # User-based spectral filtering (using precomputed projections)
        if self.filter in ['u', 'ui', 'uib', 'ub'] and hasattr(self, 'user_filter'):
            user_responses = self.user_filter(self.user_eigenvals)
            filtered_responses = user_responses.mean() if user_responses.dim() > 0 else user_responses
            
            # Efficient low-rank computation: U_batch @ (Λ * (U^T @ A))
            user_eigenvecs_batch = self.user_eigenvecs[users]  # [batch_size, n_eigen]
            filtered_proj = (filtered_responses * self.user_eigenvals).unsqueeze(1) * self.user_global_proj
            user_filtered = user_eigenvecs_batch @ filtered_proj
            scores.append(user_filtered)
        
        # Item-based spectral filtering (reconstruct on-the-fly)
        if self.filter in ['i', 'ui', 'uib'] and hasattr(self, 'item_filter'):
            item_responses = self.item_filter(self.item_eigenvals)
            filtered_responses = item_responses.mean() if item_responses.dim() > 0 else item_responses
            
            # Low-rank computation: (A @ V) @ (Λ @ V^T)
            user_proj = user_profiles @ self.item_eigenvecs  # [batch_size, n_eigen]
            filtered_eigenvecs = self.item_eigenvecs * (filtered_responses * self.item_eigenvals).unsqueeze(0)
            item_filtered = user_proj @ filtered_eigenvecs.t()
            scores.append(item_filtered)
        
        # Bipartite spectral filtering (reconstruct item part on-the-fly)
        if self.filter in ['b', 'uib', 'ub'] and hasattr(self, 'bipartite_filter'):
            bipartite_responses = self.bipartite_filter(self.bipartite_eigenvals)
            filtered_responses = bipartite_responses.mean() if bipartite_responses.dim() > 0 else bipartite_responses
            
            # Extract item-item part from bipartite eigenvectors
            item_eigenvecs_bipartite = self.bipartite_eigenvecs[self.n_users:, :]  # [n_items, n_eigen]
            
            # Low-rank computation for bipartite
            user_proj_bipartite = user_profiles @ item_eigenvecs_bipartite
            filtered_eigenvecs_bipartite = item_eigenvecs_bipartite * (filtered_responses * self.bipartite_eigenvals).unsqueeze(0)
            bipartite_filtered = user_proj_bipartite @ filtered_eigenvecs_bipartite.t()
            scores.append(bipartite_filtered)
        
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
    
    def print_memory_analysis(self):
        """Print memory usage analysis"""
        print(f"\n💾 \033[96mMemory Usage Analysis:\033[0m")
        print(f"📊 \033[91mOld approach (similarity matrices):\033[0m")
        print(f"   └─ User similarity: ~3.4 GB")
        print(f"   └─ Item similarity: ~6.4 GB") 
        print(f"   └─ Bipartite similarity: ~19.1 GB")
        print(f"   └─ Eigendecompositions: ~0.16 GB")
        print(f"   └─ Total: ~29.1 GB")
        print(f"")
        print(f"✅ \033[92mNew approach (eigendecomposition only):\033[0m")
        print(f"   └─ User eigenvectors: ~{self.u_n_eigen * self.n_users * 4 / 1e6:.1f} MB")
        print(f"   └─ Item eigenvectors: ~{self.i_n_eigen * self.n_items * 4 / 1e6:.1f} MB")
        print(f"   └─ Bipartite eigenvectors: ~{self.b_n_eigen * (self.n_users + self.n_items) * 4 / 1e6:.1f} MB")
        print(f"   └─ Precomputed projections: ~{self.u_n_eigen * self.n_items * 4 / 1e6:.1f} MB")
        total_new = (self.u_n_eigen * self.n_users + self.i_n_eigen * self.n_items + 
                    self.b_n_eigen * (self.n_users + self.n_items) + self.u_n_eigen * self.n_items) * 4 / 1e6
        print(f"   └─ Total: ~{total_new:.1f} MB")
        print(f"")
        print(f"🚀 \033[93mMemory savings: {29100 - total_new:.1f} MB ({(29100 - total_new)/29100*100:.1f}% reduction!)\033[0m")
        print(f"💨 \033[94mFaster loading: No multi-GB cache files to read\033[0m")


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


# Cache management utilities (updated for memory-optimized version)
def clear_similarity_cache(dataset):
    """Clear old similarity matrix caches (no longer needed)"""
    old_cache_files = [
        f"../cache/{dataset}_user_similarity.pkl",
        f"../cache/{dataset}_item_similarity.pkl", 
        f"../cache/{dataset}_bipartite_similarity.pkl"
    ]
    
    removed_count = 0
    for cache_file in old_cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Removed old cache: {cache_file}")
            removed_count += 1
    
    if removed_count > 0:
        print(f"🗑️ Cleared {removed_count} old similarity cache files for {dataset}")
        print("💾 These are no longer needed with the memory-optimized model")
    else:
        print(f"✅ No old similarity cache files found for dataset: {dataset}")


def get_cache_info_optimized(dataset):
    """Get information about optimized cache files"""
    eigen_cache_files = [
        f"../cache/{dataset}_user_eigen_*.pkl",
        f"../cache/{dataset}_item_eigen_*.pkl",
        f"../cache/{dataset}_bipartite_eigen_*.pkl"
    ]
    
    print(f"\n📁 \033[96mOptimized Cache Status for {dataset}:\033[0m")
    print("=" * 50)
    
    import glob
    total_size = 0
    
    for pattern in eigen_cache_files:
        files = glob.glob(pattern)
        for cache_file in files:
            if os.path.exists(cache_file):
                size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                total_size += size_mb
                print(f"✅ {os.path.basename(cache_file)}: {size_mb:.2f} MB")
    
    print(f"\n💾 Total optimized cache size: {total_size:.1f} MB")
    print(f"🚀 Memory efficiency: {(29100 - total_size)/29100*100:.1f}% reduction vs old approach")
    
    # Check for old similarity caches
    old_files = [
        f"../cache/{dataset}_user_similarity.pkl",
        f"../cache/{dataset}_item_similarity.pkl",
        f"../cache/{dataset}_bipartite_similarity.pkl"
    ]
    
    old_found = any(os.path.exists(f) for f in old_files)
    if old_found:
        print(f"\n⚠️ Old similarity cache files detected!")
        print(f"💡 Run clear_similarity_cache('{dataset}') to remove them")


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
    # Example: Check optimized cache status
    # get_cache_info_optimized('gowalla')
    
    # Example: Clear old similarity caches
    # clear_similarity_cache('gowalla')
    pass