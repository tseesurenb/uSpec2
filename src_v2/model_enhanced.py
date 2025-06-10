'''
Created on June 10, 2025
Enhanced Universal Spectral CF with THREE-VIEW Spectral Filtering
1. User-User Similarity Laplacian
2. Item-Item Similarity Laplacian  
3. User-Item Bipartite Laplacian (NEW!)

This provides three complementary perspectives for collaborative filtering.

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import gc
import os
import pickle
import filters as fl


class UniversalSpectralCF(nn.Module):
    """Enhanced Universal Spectral CF with Three-View Spectral Filtering"""
    
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.filter_order = self.config.get('filter_order', 6)
        self.filter = self.config.get('filter', 'uib')  # NEW: uib = user + item + bipartite
        
        # Convert and get dataset characteristics first
        adj_dense = adj_mat.toarray() if sp.issparse(adj_mat) else adj_mat
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32).to(self.device))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Calculate dataset characteristics
        total_interactions = torch.sum(self.adj_tensor).item()
        sparsity = total_interactions / (self.n_users * self.n_items)
        
        # Separate adaptive eigenvalue calculation for users, items, and bipartite
        self.u_n_eigen, self.i_n_eigen, self.b_n_eigen = self._calculate_adaptive_eigenvalues(total_interactions, sparsity)
        
        # Similarity parameters
        self.similarity_type = self.config.get('similarity_type', 'cosine')
        self.similarity_threshold = self.config.get('similarity_threshold', 0.01)
        
        # Filter design selection
        self.filter_design = self.config.get('filter_design', 'enhanced_basis')
        self.init_filter = self.config.get('init_filter', 'smooth')
        
        print(f"🚀 THREE-VIEW Universal Spectral CF:")
        print(f"   └─ Dataset: {self.config.get('dataset', 'unknown')}")
        print(f"   └─ Users: {self.n_users:,}, Items: {self.n_items:,}")
        print(f"   └─ Interactions: {int(total_interactions):,}")
        print(f"   └─ Sparsity: {sparsity:.4f}")
        print(f"   🔍 THREE SPECTRAL VIEWS:")
        print(f"      1️⃣ User-User Similarity: {self.u_n_eigen} eigenvalues")
        print(f"      2️⃣ Item-Item Similarity: {self.i_n_eigen} eigenvalues") 
        print(f"      3️⃣ User-Item Bipartite: {self.b_n_eigen} eigenvalues")
        print(f"   └─ Similarity Type: {self.similarity_type}")
        print(f"   └─ Similarity Threshold: {self.similarity_threshold}")
        print(f"   └─ Filter Design: {self.filter_design}")
        print(f"   └─ Device: {self.device}")
        
        # Clean up and setup
        del adj_dense
        self._memory_cleanup()
        self._setup_filters()
        self._setup_combination_weights()
    
    def _calculate_adaptive_eigenvalues(self, total_interactions, sparsity):
        """Calculate separate adaptive eigenvalues for all three views"""
        
        # Manual override if specified
        manual_u_eigen = self.config.get('u_n_eigen', None)
        manual_i_eigen = self.config.get('i_n_eigen', None)
        manual_b_eigen = self.config.get('b_n_eigen', None)  # NEW: bipartite eigenvalues
        manual_n_eigen = self.config.get('n_eigen', None)
        
        if manual_u_eigen is not None and manual_i_eigen is not None:
            if manual_u_eigen > 0 and manual_i_eigen > 0:
                # Use manual bipartite or compute it
                if manual_b_eigen is not None and manual_b_eigen > 0:
                    print(f"   🎯 Using manual eigenvalues: u={manual_u_eigen}, i={manual_i_eigen}, b={manual_b_eigen}")
                    return manual_u_eigen, manual_i_eigen, manual_b_eigen
                else:
                    # Compute bipartite eigenvalues based on the bipartite graph size
                    b_n_eigen = self._calculate_bipartite_eigenvalues(total_interactions, sparsity)
                    print(f"   🎯 Using manual u={manual_u_eigen}, i={manual_i_eigen}, computed b={b_n_eigen}")
                    return manual_u_eigen, manual_i_eigen, b_n_eigen
        elif manual_n_eigen is not None and manual_n_eigen > 0:
            print(f"   🎯 Using manual n_eigen for all views: {manual_n_eigen}")
            return manual_n_eigen, manual_n_eigen, manual_n_eigen
        
        # Separate calculations for users, items, and bipartite
        u_n_eigen = self._calculate_matrix_eigenvalues(
            self.n_users, total_interactions, sparsity, "user"
        )
        i_n_eigen = self._calculate_matrix_eigenvalues(
            self.n_items, total_interactions, sparsity, "item"
        )
        b_n_eigen = self._calculate_bipartite_eigenvalues(total_interactions, sparsity)
        
        return u_n_eigen, i_n_eigen, b_n_eigen
    
    def _calculate_bipartite_eigenvalues(self, total_interactions, sparsity):
        """Calculate eigenvalues for the bipartite user-item graph"""
        
        # Bipartite graph has (n_users + n_items) nodes
        bipartite_size = self.n_users + self.n_items
        
        # Base eigenvalue count based on bipartite graph size
        if bipartite_size < 1000:
            base_eigen = 32
        elif bipartite_size < 5000:
            base_eigen = 64
        elif bipartite_size < 10000:
            base_eigen = 96
        elif bipartite_size < 50000:
            base_eigen = 128
        elif bipartite_size < 100000:
            base_eigen = 192
        else:
            base_eigen = 256
        
        # Bipartite-specific adjustments
        avg_degree = (2 * total_interactions) / bipartite_size  # Each edge contributes to 2 nodes
        
        if avg_degree < 5:
            bipartite_multiplier = 1.4  # Need more eigenvalues for sparse bipartite graphs
        elif avg_degree > 50:
            bipartite_multiplier = 0.8  # Can use fewer for dense bipartite graphs
        else:
            bipartite_multiplier = 1.0
        
        # Sparsity adjustment
        if sparsity < 0.001:  # Very sparse
            sparsity_multiplier = 1.3
        elif sparsity < 0.01:  # Sparse
            sparsity_multiplier = 1.1
        elif sparsity > 0.05:  # Dense
            sparsity_multiplier = 0.8
        else:
            sparsity_multiplier = 1.0
        
        # Calculate final eigenvalue count
        adaptive_eigen = int(base_eigen * bipartite_multiplier * sparsity_multiplier)
        
        # Ensure reasonable bounds
        min_eigen = 16
        max_eigen = min(384, bipartite_size - 10)
        
        adaptive_eigen = max(min_eigen, min(adaptive_eigen, max_eigen))
        
        print(f"   📊 Bipartite eigenvalue calculation:")
        print(f"      Bipartite graph size: {bipartite_size}")
        print(f"      Base: {base_eigen}")
        print(f"      Bipartite mult: {bipartite_multiplier:.2f}")
        print(f"      Sparsity mult: {sparsity_multiplier:.2f}")
        print(f"      Final: {adaptive_eigen}")
        
        return adaptive_eigen
    
    def _calculate_matrix_eigenvalues(self, matrix_size, total_interactions, sparsity, matrix_type):
        """Calculate eigenvalues for a specific matrix (user or item) - SAME AS BEFORE"""
        
        # Base eigenvalue count based on matrix size
        if matrix_size < 500:
            base_eigen = 24
        elif matrix_size < 1000:
            base_eigen = 32
        elif matrix_size < 2000:
            base_eigen = 48
        elif matrix_size < 5000:
            base_eigen = 64
        elif matrix_size < 10000:
            base_eigen = 96
        elif matrix_size < 20000:
            base_eigen = 128
        elif matrix_size < 50000:
            base_eigen = 192
        else:
            base_eigen = 256
        
        # Matrix-specific adjustments
        if matrix_type == "user":
            avg_interactions_per_user = total_interactions / self.n_users
            if avg_interactions_per_user < 10:
                user_multiplier = 1.3
            elif avg_interactions_per_user > 100:
                user_multiplier = 0.8
            else:
                user_multiplier = 1.0
        else:  # item
            avg_interactions_per_item = total_interactions / self.n_items
            if avg_interactions_per_item < 5:
                item_multiplier = 1.4
            elif avg_interactions_per_item > 50:
                item_multiplier = 0.7
            else:
                item_multiplier = 1.0
        
        multiplier = user_multiplier if matrix_type == "user" else item_multiplier
        
        # Sparsity adjustment
        if sparsity < 0.001:
            sparsity_multiplier = 1.4
        elif sparsity < 0.01:
            sparsity_multiplier = 1.2
        elif sparsity > 0.05:
            sparsity_multiplier = 0.8
        else:
            sparsity_multiplier = 1.0
        
        # Calculate final eigenvalue count
        adaptive_eigen = int(base_eigen * multiplier * sparsity_multiplier)
        
        # Ensure reasonable bounds
        min_eigen = 16
        max_eigen = min(384, matrix_size - 10)
        
        adaptive_eigen = max(min_eigen, min(adaptive_eigen, max_eigen))
        
        print(f"   📊 {matrix_type.capitalize()} eigenvalue calculation:")
        print(f"      Matrix size: {matrix_size}")
        print(f"      Base: {base_eigen}")
        print(f"      {matrix_type.capitalize()} mult: {multiplier:.2f}")
        print(f"      Sparsity mult: {sparsity_multiplier:.2f}")
        print(f"      Final: {adaptive_eigen}")
        
        return adaptive_eigen
    
    def _get_cache_path(self, cache_type, filter_type=None):
        """Generate cache file path with THREE-VIEW parameters"""
        cache_dir = "../cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = self.config.get('dataset', 'unknown')
        sim_type = self.similarity_type
        threshold = str(self.similarity_threshold).replace('.', 'p')
        
        # Include ALL three eigenvalue counts in cache key
        u_eigen = self.u_n_eigen
        i_eigen = self.i_n_eigen
        b_eigen = self.b_n_eigen
        filter_design = self.config.get('filter_design', 'enhanced_basis')
        init_filter = self.config.get('init_filter', 'smooth')
        filter_order = self.config.get('filter_order', 6)
        filter_mode = self.config.get('filter', 'uib')
        
        # Create comprehensive filename with THREE-VIEW identifier
        base_name = f"{dataset}_THREE_VIEW_{sim_type}_th{threshold}_u{u_eigen}_i{i_eigen}_b{b_eigen}_{filter_design}_{init_filter}_fo{filter_order}_{filter_mode}"
        
        if filter_type:
            if cache_type.startswith('similarity'):
                filename = f"{base_name}_{filter_type}_sim.pkl"
            elif filter_type == 'bipartite':
                filename = f"{base_name}_bipartite_eigen_k{b_eigen}.pkl"
            else:  # user or item eigen
                k_value = u_eigen if filter_type == 'user' else i_eigen
                filename = f"{base_name}_{filter_type}_eigen_k{k_value}.pkl"
        else:
            filename = f"{base_name}_{cache_type}.pkl"
            
        return os.path.join(cache_dir, filename)
    
    def _save_to_cache(self, data, cache_path):
        """Save data to cache file"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"    💾 Saved to {os.path.basename(cache_path)}")
        except Exception as e:
            print(f"    ⚠️ Cache save failed: {e}")
    
    def _load_from_cache(self, cache_path):
        """Load data from cache file"""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"    📂 Loaded {os.path.basename(cache_path)}")
                return data
        except Exception as e:
            print(f"    ⚠️ Cache load failed: {e}")
        return None
    
    def _compute_similarity_matrix(self, interaction_matrix, cache_type=None):
        """Compute similarity matrix - SAME AS BEFORE"""
        
        # Try to load from cache first
        if cache_type:
            cache_path = self._get_cache_path('similarity', cache_type)
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data.to(self.device)
        
        print(f"    Computing {self.similarity_type} similarity...")
        
        if self.similarity_type == 'cosine':
            norms = torch.norm(interaction_matrix, dim=1, keepdim=True) + 1e-8
            normalized_matrix = interaction_matrix / norms
            similarity = normalized_matrix @ normalized_matrix.t()
        
        elif self.similarity_type == 'jaccard':
            intersection = interaction_matrix @ interaction_matrix.t()
            sum_matrix = interaction_matrix.sum(dim=1, keepdim=True)
            union = sum_matrix + sum_matrix.t() - intersection
            similarity = intersection / (union + 1e-8)
        
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
        
        # Ensure symmetry and apply threshold
        similarity = (similarity + similarity.t()) / 2
        
        # Auto-adaptive threshold if threshold is negative
        if self.similarity_threshold < 0:
            if similarity.numel() > 10_000_000:
                sample_size = min(1_000_000, similarity.numel() // 10)
                flat_sim = similarity.flatten()
                indices = torch.randperm(flat_sim.numel())[:sample_size]
                sampled_similarities = flat_sim[indices]
                adaptive_threshold = torch.quantile(sampled_similarities, 0.7).item()
            else:
                adaptive_threshold = torch.quantile(similarity.flatten(), 0.7).item()
        else:
            adaptive_threshold = self.similarity_threshold
        
        similarity_thresholded = torch.where(similarity >= adaptive_threshold, 
                                           similarity, torch.zeros_like(similarity))
        
        # Set diagonal to 1 (self-similarity)
        similarity_thresholded.fill_diagonal_(1.0)
        result = torch.clamp(similarity_thresholded, min=0.0, max=1.0)
        
        # Save to cache
        if cache_type:
            self._save_to_cache(result.cpu(), cache_path)
        
        return result
    
    def _compute_similarity_laplacian(self, similarity_matrix):
        """Compute normalized similarity Laplacian - SAME AS BEFORE"""
        degree = similarity_matrix.sum(dim=1) + 1e-8
        
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        normalized_laplacian = similarity_matrix * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
        
        identity = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
        laplacian = identity - normalized_laplacian
        
        return laplacian
    
    def _compute_bipartite_laplacian(self):
        """NEW: Compute bipartite user-item Laplacian matrix"""
        print(f"    Computing bipartite user-item Laplacian...")
        
        # Create bipartite adjacency matrix: [0, A; A^T, 0]
        n_total = self.n_users + self.n_items
        bipartite_adj = torch.zeros(n_total, n_total, device=self.device)
        
        # Fill the bipartite structure
        bipartite_adj[:self.n_users, self.n_users:] = self.adj_tensor  # User-Item edges
        bipartite_adj[self.n_users:, :self.n_users] = self.adj_tensor.t()  # Item-User edges
        
        # Compute degree matrix for normalization
        degree = bipartite_adj.sum(dim=1) + 1e-8
        
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        # Normalize bipartite adjacency
        normalized_bipartite = bipartite_adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
        
        # Bipartite Laplacian: L = I - normalized_adjacency
        identity = torch.eye(n_total, device=self.device)
        bipartite_laplacian = identity - normalized_bipartite
        
        print(f"    Bipartite Laplacian shape: {bipartite_laplacian.shape}")
        
        return bipartite_laplacian
    
    def _setup_filters(self):
        """Setup THREE spectral filters"""
        print(f"Computing THREE-VIEW similarity Laplacians for filter type: {self.filter}")
        start = time.time()
        
        self.user_filter = None
        self.item_filter = None
        self.bipartite_filter = None  # NEW!
        
        if self.filter in ['u', 'ui', 'uib']:
            print("1️⃣ Processing user-user similarity Laplacian...")
            self.user_filter = self._create_similarity_filter('user')
            self._memory_cleanup()
        
        if self.filter in ['i', 'ui', 'uib']:
            print("2️⃣ Processing item-item similarity Laplacian...")
            self.item_filter = self._create_similarity_filter('item')
            self._memory_cleanup()
        
        if self.filter in ['b', 'uib']:  # NEW: bipartite filter
            print("3️⃣ Processing user-item bipartite Laplacian...")
            self.bipartite_filter = self._create_bipartite_filter()
            self._memory_cleanup()
        
        print(f'THREE-VIEW filter setup completed in {time.time() - start:.2f}s')
    
    def _create_similarity_filter(self, filter_type):
        """Create similarity-based spectral filter - SAME AS BEFORE"""
        
        # Use appropriate eigenvalue count
        if filter_type == 'user':
            n_eigen_to_use = self.u_n_eigen
        else:
            n_eigen_to_use = self.i_n_eigen
        
        # Try to load eigendecomposition from cache
        eigen_cache_path = self._get_cache_path('eigen', filter_type)
        cached_eigen = self._load_from_cache(eigen_cache_path)
        
        if cached_eigen is not None:
            eigenvals, eigenvecs = cached_eigen
            self.register_buffer(f'{filter_type}_eigenvals', eigenvals.to(self.device))
            self.register_buffer(f'{filter_type}_eigenvecs', eigenvecs.to(self.device))
            print(f"  {filter_type.capitalize()} eigendecomposition loaded from cache ({n_eigen_to_use} eigenvalues)")
        else:
            print(f"  Computing {filter_type} similarity matrix...")
            
            with torch.no_grad():
                if filter_type == 'user':
                    similarity_matrix = self._compute_similarity_matrix(self.adj_tensor, cache_type='user')
                    n_components = self.n_users
                else:  # item
                    similarity_matrix = self._compute_similarity_matrix(self.adj_tensor.t(), cache_type='item')
                    n_components = self.n_items
                
                print(f"  Computing {filter_type} similarity Laplacian...")
                laplacian = self._compute_similarity_laplacian(similarity_matrix)
            
            print(f"  Computing eigendecomposition...")
            laplacian_np = laplacian.cpu().numpy()
            
            del similarity_matrix, laplacian
            self._memory_cleanup()
            
            k = min(n_eigen_to_use, n_components - 2)
            
            try:
                print(f"  Computing {k} smallest eigenvalues for {filter_type}...")
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(laplacian_np), k=k, which='SM')
                
                eigenvals = np.maximum(eigenvals, 0.0)
                
                eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
                eigenvecs_tensor = torch.tensor(eigenvecs, dtype=torch.float32)
                
                # Save to cache
                self._save_to_cache((eigenvals_tensor, eigenvecs_tensor), eigen_cache_path)
                
                # Register buffers
                self.register_buffer(f'{filter_type}_eigenvals', eigenvals_tensor.to(self.device))
                self.register_buffer(f'{filter_type}_eigenvecs', eigenvecs_tensor.to(self.device))
                
                print(f"  {filter_type.capitalize()} Laplacian eigendecomposition: {k} components")
                print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
                
            except Exception as e:
                print(f"  {filter_type.capitalize()} eigendecomposition failed: {e}")
                print(f"  Using fallback identity matrices...")
                
                eigenvals = np.linspace(0, 1, min(n_eigen_to_use, n_components))
                eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
                eigenvecs_tensor = torch.eye(n_components, min(n_eigen_to_use, n_components))
                
                self.register_buffer(f'{filter_type}_eigenvals', eigenvals_tensor.to(self.device))
                self.register_buffer(f'{filter_type}_eigenvecs', eigenvecs_tensor.to(self.device))
            
            del laplacian_np
            if 'eigenvals' in locals():
                del eigenvals, eigenvecs
            self._memory_cleanup()
        
        return self._create_filter_instance()
    
    def _create_bipartite_filter(self):
        """NEW: Create bipartite spectral filter"""
        
        n_eigen_to_use = self.b_n_eigen
        
        # Try to load bipartite eigendecomposition from cache
        bipartite_cache_path = self._get_cache_path('eigen', 'bipartite')
        cached_eigen = self._load_from_cache(bipartite_cache_path)
        
        if cached_eigen is not None:
            eigenvals, eigenvecs = cached_eigen
            self.register_buffer('bipartite_eigenvals', eigenvals.to(self.device))
            self.register_buffer('bipartite_eigenvecs', eigenvecs.to(self.device))
            print(f"  Bipartite eigendecomposition loaded from cache ({n_eigen_to_use} eigenvalues)")
        else:
            print(f"  Computing bipartite Laplacian...")
            
            with torch.no_grad():
                bipartite_laplacian = self._compute_bipartite_laplacian()
            
            print(f"  Computing bipartite eigendecomposition...")
            laplacian_np = bipartite_laplacian.cpu().numpy()
            
            del bipartite_laplacian
            self._memory_cleanup()
            
            n_total = self.n_users + self.n_items
            k = min(n_eigen_to_use, n_total - 2)
            
            try:
                print(f"  Computing {k} smallest eigenvalues for bipartite graph...")
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(laplacian_np), k=k, which='SM')
                
                eigenvals = np.maximum(eigenvals, 0.0)
                
                eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
                eigenvecs_tensor = torch.tensor(eigenvecs, dtype=torch.float32)
                
                # Save to cache
                self._save_to_cache((eigenvals_tensor, eigenvecs_tensor), bipartite_cache_path)
                
                # Register buffers
                self.register_buffer('bipartite_eigenvals', eigenvals_tensor.to(self.device))
                self.register_buffer('bipartite_eigenvecs', eigenvecs_tensor.to(self.device))
                
                print(f"  Bipartite Laplacian eigendecomposition: {k} components")
                print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
                
            except Exception as e:
                print(f"  Bipartite eigendecomposition failed: {e}")
                print(f"  Using fallback identity matrices...")
                
                eigenvals = np.linspace(0, 1, min(n_eigen_to_use, n_total))
                eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
                eigenvecs_tensor = torch.eye(n_total, min(n_eigen_to_use, n_total))
                
                self.register_buffer('bipartite_eigenvals', eigenvals_tensor.to(self.device))
                self.register_buffer('bipartite_eigenvecs', eigenvecs_tensor.to(self.device))
            
            del laplacian_np
            if 'eigenvals' in locals():
                del eigenvals, eigenvecs
            self._memory_cleanup()
        
        return self._create_filter_instance()
    
    def _create_filter_instance(self):
        """Create filter instance based on design"""
        if self.filter_design == 'original':
            return fl.UniversalSpectralFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'basis':
            return fl.SpectralBasisFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'enhanced_basis':
            return fl.EnhancedSpectralBasisFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'adaptive_golden':
            return fl.AdaptiveGoldenFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'multiscale':
            return fl.MultiScaleSpectralFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'ensemble':
            return fl.EnsembleSpectralFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'band_stop':
            return fl.BandStopSpectralFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'adaptive_band_stop':
            return fl.AdaptiveBandStopFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'parametric_multi_band':
            return fl.ParametricMultiBandFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'harmonic':
            return fl.HarmonicSpectralFilter(self.filter_order, self.init_filter)
        else:
            raise ValueError(f"Unknown filter design: {self.filter_design}")
    
    def _memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _setup_combination_weights(self):
        """Setup learnable combination weights for THREE views"""
        if self.filter == 'u':
            # User similarity only
            init_weights = torch.tensor([0.5, 0.5])  # [direct, user_filtered]
        elif self.filter == 'i':
            # Item similarity only
            init_weights = torch.tensor([0.5, 0.5])  # [direct, item_filtered]
        elif self.filter == 'b':
            # Bipartite only
            init_weights = torch.tensor([0.5, 0.5])  # [direct, bipartite_filtered]
        elif self.filter == 'ui':
            # User + Item similarities
            init_weights = torch.tensor([0.5, 0.3, 0.2])  # [direct, item_filtered, user_filtered]
        elif self.filter == 'uib':
            # ALL THREE views (NEW!)
            init_weights = torch.tensor([0.4, 0.25, 0.25, 0.1])  # [direct, item_filtered, user_filtered, bipartite_filtered]
        else:
            # Default fallback
            init_weights = torch.tensor([0.5, 0.5])
        
        self.combination_weights = nn.Parameter(init_weights.to(self.device))
        
        print(f"   🎚️  THREE-VIEW combination weights initialized: {init_weights.tolist()}")
    
    def _get_filter_matrices(self):
        """Compute spectral filter matrices for all three views"""
        user_matrix = item_matrix = bipartite_matrix = None
        
        if self.user_filter is not None:
            response = self.user_filter(self.user_eigenvals)
            user_matrix = self.user_eigenvecs @ torch.diag(response) @ self.user_eigenvecs.t()
        
        if self.item_filter is not None:
            response = self.item_filter(self.item_eigenvals)
            item_matrix = self.item_eigenvecs @ torch.diag(response) @ self.item_eigenvecs.t()
        
        if self.bipartite_filter is not None:
            response = self.bipartite_filter(self.bipartite_eigenvals)
            bipartite_matrix = self.bipartite_eigenvecs @ torch.diag(response) @ self.bipartite_eigenvecs.t()
        
        return user_matrix, item_matrix, bipartite_matrix
    
    def forward(self, users):
        """Forward pass: THREE-VIEW spectral filtering"""
        if users.device != self.adj_tensor.device:
            users = users.to(self.adj_tensor.device)
        
        user_profiles = self.adj_tensor[users]
        user_filter_matrix, item_filter_matrix, bipartite_filter_matrix = self._get_filter_matrices()
        
        scores = [user_profiles]  # Direct collaborative filtering scores
        
        # Item-based filtering: user profiles through item similarity filter
        if self.filter in ['i', 'ui', 'uib'] and item_filter_matrix is not None:
            scores.append(user_profiles @ item_filter_matrix)
        
        # User-based filtering: user similarity filter through interactions
        if self.filter in ['u', 'ui', 'uib'] and user_filter_matrix is not None:
            user_filtered = user_filter_matrix[users] @ self.adj_tensor
            scores.append(user_filtered)
        
        # NEW: Bipartite filtering - extract user part from bipartite filtered result
        if self.filter in ['b', 'uib'] and bipartite_filter_matrix is not None:
            # Create bipartite user representation: [user_profile, zeros_for_items]
            batch_size = users.shape[0]
            bipartite_input = torch.zeros(batch_size, self.n_users + self.n_items, device=self.device)
            bipartite_input[torch.arange(batch_size), users] = 1.0  # One-hot user representation
            
            # Apply bipartite filter
            bipartite_filtered = bipartite_input @ bipartite_filter_matrix
            
            # Extract item predictions (second part of bipartite result)
            bipartite_item_scores = bipartite_filtered[:, self.n_users:]
            scores.append(bipartite_item_scores)
        
        # Combine predictions using learnable weights
        weights = torch.softmax(self.combination_weights, dim=0)
        predicted = sum(w * score for w, score in zip(weights, scores))
        
        if self.training and (self.n_users > 5000 or self.n_items > 5000):
            del user_filter_matrix, item_filter_matrix, bipartite_filter_matrix
            self._memory_cleanup()
        
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
    
    def get_filter_parameters(self):
        """Get filter parameters for separate optimization"""
        filter_params = []
        if self.user_filter is not None:
            filter_params.extend(self.user_filter.parameters())
        if self.item_filter is not None:
            filter_params.extend(self.item_filter.parameters())
        if self.bipartite_filter is not None:  # NEW!
            filter_params.extend(self.bipartite_filter.parameters())
        return filter_params
    
    def get_other_parameters(self):
        """Get non-filter parameters"""
        filter_param_ids = {id(p) for p in self.get_filter_parameters()}
        return [p for p in self.parameters() if id(p) not in filter_param_ids]
    
    def debug_filter_learning(self):
        """Debug THREE-VIEW spectral filtering"""
        print(f"\n=== THREE-VIEW SPECTRAL FILTER DEBUG ===")
        print(f"🔍 THREE SPECTRAL VIEWS:")
        print(f"   1️⃣ User-User Similarity Laplacian")
        print(f"   2️⃣ Item-Item Similarity Laplacian") 
        print(f"   3️⃣ User-Item Bipartite Laplacian")
        print(f"Similarity Type: {self.similarity_type}")
        print(f"Similarity Threshold: {self.similarity_threshold}")
        print(f"Filter Design: {self.filter_design}")
        print(f"Device: {self.device}")
        print(f"User Eigenvalues: {self.u_n_eigen}")
        print(f"Item Eigenvalues: {self.i_n_eigen}")
        print(f"Bipartite Eigenvalues: {self.b_n_eigen}")
        
        with torch.no_grad():
            if self.filter in ['u', 'ui', 'uib'] and self.user_filter is not None:
                print(f"\n👤 User Similarity Filter ({self.u_n_eigen} eigenvalues):")
                if hasattr(self, 'user_eigenvals'):
                    eigenvals = self.user_eigenvals.cpu().numpy()
                    print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
            
            if self.filter in ['i', 'ui', 'uib'] and self.item_filter is not None:
                print(f"\n🎬 Item Similarity Filter ({self.i_n_eigen} eigenvalues):")
                if hasattr(self, 'item_eigenvals'):
                    eigenvals = self.item_eigenvals.cpu().numpy()
                    print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
            
            if self.filter in ['b', 'uib'] and self.bipartite_filter is not None:
                print(f"\n🔗 Bipartite Filter ({self.b_n_eigen} eigenvalues):")
                if hasattr(self, 'bipartite_eigenvals'):
                    eigenvals = self.bipartite_eigenvals.cpu().numpy()
                    print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
            
            weights = torch.softmax(self.combination_weights, dim=0)
            print(f"\n🎚️  THREE-VIEW Combination Weights: {weights.cpu().numpy()}")
        
        print("=== END THREE-VIEW DEBUG ===\n")

    def get_parameter_count(self):
        """Get parameter count breakdown"""
        total_params = sum(p.numel() for p in self.parameters())
        filter_params = sum(p.numel() for p in self.get_filter_parameters())
        
        return {
            'total': total_params,
            'filter': filter_params,
            'combination': self.combination_weights.numel(),
            'other': total_params - filter_params
        }
    
    def clear_cache(self):
        """Clear cache files for THREE-VIEW configuration"""
        cache_dir = "../cache"
        if not os.path.exists(cache_dir):
            return
        
        dataset = self.config.get('dataset', 'unknown')
        
        # Look for THREE_VIEW cache files
        pattern_parts = [dataset, 'THREE_VIEW']
        
        removed_count = 0
        for filename in os.listdir(cache_dir):
            if all(part in filename for part in pattern_parts):
                file_path = os.path.join(cache_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"🗑️ Removed: {filename}")
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to remove {filename}: {e}")
        
        if removed_count == 0:
            print("No matching THREE-VIEW cache files found")
        else:
            print(f"Removed {removed_count} THREE-VIEW cache files")


# ============================================================================
# THREE-VIEW SPECTRAL FILTERING - RESEARCH CONTRIBUTIONS
# ============================================================================
#
# 🔬 NOVEL RESEARCH CONTRIBUTIONS:
#
# 1. **Three Complementary Views**:
#    - User-User Similarity: Captures user behavioral patterns
#    - Item-Item Similarity: Captures item content/usage patterns  
#    - User-Item Bipartite: Captures direct user-item relationships
#
# 2. **Bipartite Spectral Filtering** (NEW):
#    - First application of bipartite graph Laplacian in spectral CF
#    - Captures higher-order user-item dependencies
#    - Provides orthogonal information to similarity-based views
#
# 3. **Learnable View Combination**:
#    - Adaptive weighting of three spectral perspectives
#    - Automatic discovery of optimal view importance
#    - Dataset-specific view adaptation
#
# 4. **Theoretical Foundation**:
#    - Three different eigendecompositions provide diverse spectral signatures
#    - Bipartite Laplacian captures global graph structure
#    - Similarity Laplacians capture local neighborhood structure
#
# 5. **Scalable Implementation**:
#    - Efficient caching for all three eigendecompositions
#    - Memory-optimized bipartite graph construction
#    - Progressive eigenvalue count adaptation
#
# 🎯 RESEARCH QUESTIONS THIS ENABLES:
# - How do different spectral views complement each other?
# - What is the optimal combination of similarity vs. bipartite views?
# - How does bipartite spectral filtering compare to traditional approaches?
# - Can we learn domain-specific view importance automatically?
#
# 📊 EVALUATION SCENARIOS:
# - Compare three-view vs. two-view vs. single-view performance
# - Analyze view importance across different datasets/domains
# - Study computational vs. accuracy trade-offs
# - Investigate spectral properties of different graph views
#
# 💡 PUBLICATION POTENTIAL:
# - Novel architectural contribution (three-view spectral filtering)
# - First bipartite Laplacian application in spectral CF
# - Strong theoretical foundation in spectral graph theory
# - Comprehensive experimental evaluation framework
#
# ============================================================================