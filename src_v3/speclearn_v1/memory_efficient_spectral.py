"""
Memory-efficient spectral operations for large datasets
Maintains all views while handling Amazon-book scale
"""
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import gc
import psutil


def get_memory_usage():
    """Get current memory usage in GB"""
    return psutil.Process().memory_info().rss / 1024 / 1024 / 1024


def compute_similarity_chunked(adj_mat, view='item', chunk_size=10000):
    """
    Compute similarity matrix in chunks to control memory usage
    Maintains sparsity throughout computation
    """
    n_users, n_items = adj_mat.shape
    
    # Normalize adjacency matrix
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum + 1e-10, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_u = sp.diags(d_inv)
    
    colsum = np.array(adj_mat.sum(axis=0)) 
    d_inv = np.power(colsum + 1e-10, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_i = sp.diags(d_inv)
    
    norm_adj = d_mat_u.dot(adj_mat).dot(d_mat_i)
    
    if view == 'user':
        # User similarity: norm_adj @ norm_adj.T
        # Process in chunks
        n = n_users
        similarity = sp.lil_matrix((n, n), dtype=np.float32)
        
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            print(f"Processing user chunk {i}-{end_i} / {n}, Memory: {get_memory_usage():.2f}GB")
            
            # Compute chunk of similarity
            chunk = norm_adj[i:end_i] @ norm_adj.T
            
            # Sparsify: keep only top-k per row
            for j in range(chunk.shape[0]):
                row = chunk[j].toarray().flatten()
                # Keep top 500 similar users
                top_k = min(500, n)
                top_idx = np.argpartition(row, -top_k)[-top_k:]
                
                # Store in sparse format
                for idx in top_idx:
                    if row[idx] > 1e-6:  # Threshold for numerical stability
                        similarity[i+j, idx] = row[idx]
            
            # Force garbage collection
            del chunk
            gc.collect()
        
        return similarity.tocsr()
    
    elif view == 'item':
        # Item similarity: norm_adj.T @ norm_adj
        # For very large item sets, use randomized approach
        if n_items > 100000:
            print(f"Using randomized similarity for {n_items} items")
            return compute_randomized_similarity(norm_adj.T, n_items)
        else:
            # Standard chunked computation
            n = n_items
            similarity = sp.lil_matrix((n, n), dtype=np.float32)
            
            for i in range(0, n, chunk_size):
                end_i = min(i + chunk_size, n)
                print(f"Processing item chunk {i}-{end_i} / {n}, Memory: {get_memory_usage():.2f}GB")
                
                # Compute chunk
                chunk = norm_adj.T[i:end_i] @ norm_adj
                
                # Sparsify
                for j in range(chunk.shape[0]):
                    row = chunk[j].toarray().flatten()
                    top_k = min(1000, n)  # More neighbors for items
                    top_idx = np.argpartition(row, -top_k)[-top_k:]
                    
                    for idx in top_idx:
                        if row[idx] > 1e-6:
                            similarity[i+j, idx] = row[idx]
                
                del chunk
                gc.collect()
            
            return similarity.tocsr()


def compute_randomized_similarity(norm_adj_T, n_items, rank=1000):
    """
    Use randomized SVD to approximate similarity for very large matrices
    This avoids computing the full n_items x n_items similarity matrix
    """
    from sklearn.utils.extmath import randomized_svd
    
    print(f"Computing randomized similarity approximation with rank {rank}")
    
    # Compute randomized SVD of normalized adjacency
    U, s, Vt = randomized_svd(norm_adj_T, n_components=rank, random_state=42)
    
    # Approximate similarity: U @ diag(s^2) @ U.T
    # But we keep it factorized for memory efficiency
    return U, s**2


def sparse_eigsh_with_restart(similarity, k=100, max_attempts=3):
    """
    Robust eigendecomposition with restart on memory error
    """
    for attempt in range(max_attempts):
        try:
            print(f"Eigendecomposition attempt {attempt+1}, k={k}")
            
            # Use shift-invert mode for better convergence on sparse matrices
            eigenvals, eigenvecs = eigsh(
                similarity, 
                k=k, 
                which='LM',
                maxiter=1000,
                tol=1e-4
            )
            
            return eigenvals, eigenvecs
            
        except MemoryError:
            print(f"Memory error, reducing k from {k} to {k//2}")
            k = k // 2
            gc.collect()
            
        except Exception as e:
            print(f"Error in eigendecomposition: {e}")
            if attempt < max_attempts - 1:
                k = int(k * 0.7)
                gc.collect()
            else:
                raise


def compute_bipartite_chunked(norm_adj, n_users, n_items, chunk_size=5000):
    """
    Compute bipartite similarity in chunks
    """
    total_size = n_users + n_items
    
    # Create block structure efficiently
    print(f"Creating bipartite structure for {total_size} nodes")
    
    # We don't need full bipartite matrix - just eigendecomposition
    # Use implicitly restarted Arnoldi method
    
    class BipartiteOperator:
        """Linear operator for bipartite matrix without explicit construction"""
        def __init__(self, norm_adj):
            self.norm_adj = norm_adj
            self.n_users = norm_adj.shape[0]
            self.n_items = norm_adj.shape[1]
            self.shape = (self.n_users + self.n_items, self.n_users + self.n_items)
            
        def _matvec(self, x):
            """Matrix-vector product for [0, A; A.T, 0] @ x"""
            x_u = x[:self.n_users]
            x_i = x[self.n_users:]
            
            y_u = self.norm_adj @ x_i
            y_i = self.norm_adj.T @ x_u
            
            return np.concatenate([y_u, y_i])
    
    # Use operator for eigendecomposition
    from scipy.sparse.linalg import LinearOperator
    
    bipartite_op = LinearOperator(
        shape=(total_size, total_size),
        matvec=BipartiteOperator(norm_adj)._matvec,
        dtype=np.float32
    )
    
    return bipartite_op


class MemoryEfficientSpectralCF:
    """Memory-efficient wrapper for large-scale spectral CF"""
    
    def __init__(self, original_model):
        self.model = original_model
        
    def setup_spectral_filters_efficient(self):
        """Override setup for memory efficiency"""
        if self.model.dataset == 'amazon-book':
            print("Using memory-efficient setup for Amazon-book")
            
            # Process each view separately with memory management
            if 'u' in self.model.filter_views:
                self._setup_user_view_efficient()
                
            if 'i' in self.model.filter_views:
                self._setup_item_view_efficient()
                
            if 'b' in self.model.filter_views:
                self._setup_bipartite_view_efficient()
        else:
            # Use standard setup for smaller datasets
            self.model._setup_spectral_filters()
    
    def _setup_user_view_efficient(self):
        """Efficient user view setup"""
        print("Setting up user view with memory optimization...")
        
        # Compute similarity in chunks
        user_sim = compute_similarity_chunked(
            self.model.adj_mat, 
            view='user',
            chunk_size=5000
        )
        
        # Eigendecomposition with restart
        k = min(self.model.u_n_eigen, user_sim.shape[0] - 1)
        eigenvals, eigenvecs = sparse_eigsh_with_restart(user_sim, k=k)
        
        self.model.register_buffer(
            'user_eigenvals', 
            torch.tensor(eigenvals, dtype=torch.float32)
        )
        self.model.register_buffer(
            'user_eigenvecs',
            torch.tensor(eigenvecs, dtype=torch.float32)
        )
        
        # Clear similarity matrix from memory
        del user_sim
        gc.collect()
        
    def _setup_item_view_efficient(self):
        """Efficient item view setup"""
        print("Setting up item view with memory optimization...")
        
        if self.model.n_items > 100000:
            # Use randomized approximation
            U, s_squared = compute_randomized_similarity(
                self.model.adj_mat.T,
                self.model.n_items
            )
            
            # Use approximate eigenvalues
            k = min(self.model.i_n_eigen, len(s_squared))
            self.model.register_buffer(
                'item_eigenvals',
                torch.tensor(s_squared[:k], dtype=torch.float32)
            )
            self.model.register_buffer(
                'item_eigenvecs',
                torch.tensor(U[:, :k], dtype=torch.float32)
            )
        else:
            # Standard chunked computation
            item_sim = compute_similarity_chunked(
                self.model.adj_mat,
                view='item', 
                chunk_size=10000
            )
            
            k = min(self.model.i_n_eigen, item_sim.shape[0] - 1)
            eigenvals, eigenvecs = sparse_eigsh_with_restart(item_sim, k=k)
            
            self.model.register_buffer(
                'item_eigenvals',
                torch.tensor(eigenvals, dtype=torch.float32)
            )
            self.model.register_buffer(
                'item_eigenvecs',
                torch.tensor(eigenvecs, dtype=torch.float32)
            )
            
            del item_sim
            gc.collect()
    
    def _setup_bipartite_view_efficient(self):
        """Efficient bipartite view setup"""
        print("Setting up bipartite view with memory optimization...")
        
        # Normalize adjacency
        rowsum = np.array(self.model.adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-10, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_u = sp.diags(d_inv)
        
        colsum = np.array(self.model.adj_mat.sum(axis=0))
        d_inv = np.power(colsum + 1e-10, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_i = sp.diags(d_inv)
        
        norm_adj = d_mat_u.dot(self.model.adj_mat).dot(d_mat_i)
        
        # Create bipartite operator
        bipartite_op = compute_bipartite_chunked(
            norm_adj,
            self.model.n_users,
            self.model.n_items
        )
        
        # Eigendecomposition using operator
        k = min(self.model.b_n_eigen, self.model.n_users + self.model.n_items - 1)
        eigenvals, eigenvecs = eigsh(bipartite_op, k=k, which='LM')
        
        self.model.register_buffer(
            'bipartite_eigenvals',
            torch.tensor(eigenvals, dtype=torch.float32)
        )
        self.model.register_buffer(
            'bipartite_eigenvecs',
            torch.tensor(eigenvecs, dtype=torch.float32)
        )
        
        gc.collect()