'''
Created on June 10, 2025
Enhanced Universal Spectral CF with THREE-VIEW Spectral Filtering - FIXED VERSION
UPDATED: Full support for 'ub' filter (User + Bipartite)
FIXED: Scaling inconsistency, eigenvalue domain confusion, combination weights

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
    """FIXED Enhanced Universal Spectral CF with Three-View Spectral Filtering"""
    
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
        
        print(f"🚀 THREE-VIEW Universal Spectral CF (FIXED):")
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
        print(f"   └─ Filter Type: {self.filter}")
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
        
        return adaptive_eigen
    
    def _calculate_matrix_eigenvalues(self, matrix_size, total_interactions, sparsity, matrix_type):
        """Calculate eigenvalues for a specific matrix (user or item)"""
        
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
        base_name = f"{dataset}_THREE_VIEW_FIXED_{sim_type}_th{threshold}_u{u_eigen}_i{i_eigen}_b{b_eigen}_{filter_design}_{init_filter}_fo{filter_order}_{filter_mode}"
        
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
            pass
        return None
    
    def _compute_similarity_matrix(self, interaction_matrix, cache_type=None):
        """Compute similarity matrix with caching"""
        
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
    
    def _compute_bipartite_laplacian(self):
        """Compute bipartite user-item Laplacian matrix"""
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
        
        # FIXED: Return normalized adjacency instead of Laplacian for similarity eigenvalues
        print(f"    Bipartite normalized adjacency shape: {normalized_bipartite.shape}")
        
        return normalized_bipartite
    
    def _setup_filters(self):
        """Setup THREE spectral filters"""
        print(f"Computing THREE-VIEW similarity matrices for filter type: {self.filter}")
        start = time.time()
        
        self.user_filter = None
        self.item_filter = None
        self.bipartite_filter = None
        
        if self.filter in ['u', 'ui', 'uib', 'ub']:
            print("1️⃣ Processing user-user similarity matrix...")
            self.user_filter = self._create_similarity_filter('user')
            self._memory_cleanup()
        
        if self.filter in ['i', 'ui', 'uib']:
            print("2️⃣ Processing item-item similarity matrix...")
            self.item_filter = self._create_similarity_filter('item')
            self._memory_cleanup()
        
        if self.filter in ['b', 'uib', 'ub']:
            print("3️⃣ Processing user-item bipartite matrix...")
            self.bipartite_filter = self._create_bipartite_filter()
            self._memory_cleanup()
        
        print(f'THREE-VIEW filter setup completed in {time.time() - start:.2f}s')
    
    def _create_similarity_filter(self, filter_type):
        """FIXED: Create similarity-based spectral filter using similarity matrices"""
        
        if filter_type == 'user':
            n_eigen_to_use = self.u_n_eigen
            n_components = self.n_users
        else:
            n_eigen_to_use = self.i_n_eigen
            n_components = self.n_items
        
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
                else:  # item
                    similarity_matrix = self._compute_similarity_matrix(self.adj_tensor.t(), cache_type='item')
            
            print(f"  Computing eigendecomposition of SIMILARITY matrix...")
            similarity_np = similarity_matrix.cpu().numpy()
            
            del similarity_matrix
            self._memory_cleanup()
            
            k = min(n_eigen_to_use, n_components - 2)
            
            try:
                print(f"  Computing {k} LARGEST eigenvalues for {filter_type} similarity...")
                # FIXED: Use 'LM' (largest magnitude) for similarity matrices
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(similarity_np), k=k, which='LM')
                
                # FIXED: For similarity matrices, eigenvalues should be positive and ≤ 1
                eigenvals = np.clip(eigenvals, 0.0, 1.0)
                
                eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
                eigenvecs_tensor = torch.tensor(eigenvecs, dtype=torch.float32)
                
                # Save to cache
                self._save_to_cache((eigenvals_tensor, eigenvecs_tensor), eigen_cache_path)
                
                # Register buffers
                self.register_buffer(f'{filter_type}_eigenvals', eigenvals_tensor.to(self.device))
                self.register_buffer(f'{filter_type}_eigenvecs', eigenvecs_tensor.to(self.device))
                
                print(f"  {filter_type.capitalize()} similarity eigendecomposition: {k} components")
                print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
                
            except Exception as e:
                print(f"  {filter_type.capitalize()} eigendecomposition failed: {e}")
                print(f"  Using fallback similarity eigenvalues...")
                
                # FIXED: Fallback for similarity matrices (decreasing from 1.0)
                eigenvals = np.linspace(1.0, 0.1, min(n_eigen_to_use, n_components))
                eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
                eigenvecs_tensor = torch.eye(n_components, min(n_eigen_to_use, n_components))
                
                self.register_buffer(f'{filter_type}_eigenvals', eigenvals_tensor.to(self.device))
                self.register_buffer(f'{filter_type}_eigenvecs', eigenvecs_tensor.to(self.device))
            
            del similarity_np
            if 'eigenvals' in locals():
                del eigenvals, eigenvecs
            self._memory_cleanup()
        
        return self._create_filter_instance()
    
    def _create_bipartite_filter(self):
        """Create bipartite spectral filter"""
        
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
            print(f"  Computing bipartite normalized adjacency...")
            
            with torch.no_grad():
                bipartite_normalized = self._compute_bipartite_laplacian()
            
            print(f"  Computing bipartite eigendecomposition...")
            normalized_np = bipartite_normalized.cpu().numpy()
            
            del bipartite_normalized
            self._memory_cleanup()
            
            n_total = self.n_users + self.n_items
            k = min(n_eigen_to_use, n_total - 2)
            
            try:
                print(f"  Computing {k} LARGEST eigenvalues for bipartite similarity...")
                # FIXED: Use 'LM' for normalized adjacency (similarity-like)
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(normalized_np), k=k, which='LM')
                
                eigenvals = np.clip(eigenvals, 0.0, 1.0)
                
                eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
                eigenvecs_tensor = torch.tensor(eigenvecs, dtype=torch.float32)
                
                # Save to cache
                self._save_to_cache((eigenvals_tensor, eigenvecs_tensor), bipartite_cache_path)
                
                # Register buffers
                self.register_buffer('bipartite_eigenvals', eigenvals_tensor.to(self.device))
                self.register_buffer('bipartite_eigenvecs', eigenvecs_tensor.to(self.device))
                
                print(f"  Bipartite eigendecomposition: {k} components")
                print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
                
            except Exception as e:
                print(f"  Bipartite eigendecomposition failed: {e}")
                print(f"  Using fallback identity matrices...")
                
                eigenvals = np.linspace(1.0, 0.1, min(n_eigen_to_use, n_total))
                eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
                eigenvecs_tensor = torch.eye(n_total, min(n_eigen_to_use, n_total))
                
                self.register_buffer('bipartite_eigenvals', eigenvals_tensor.to(self.device))
                self.register_buffer('bipartite_eigenvecs', eigenvecs_tensor.to(self.device))
            
            del normalized_np
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
        elif self.filter_design == 'chebyshev':
            return fl.ChebyshevSpectralFilter(self.filter_order, self.init_filter)
        elif self.filter_design == 'jacobi':
            polynomial_params = self.config.get('polynomial_params', {})
            alpha = polynomial_params.get('alpha', 0.0)
            beta = polynomial_params.get('beta', 0.0)
            return fl.JacobiSpectralFilter(self.filter_order, self.init_filter, alpha, beta)
        elif self.filter_design == 'legendre':
            return fl.LegendreSpectralFilter(self.filter_order, self.init_filter)
        else:
            raise ValueError(f"Unknown filter design: {self.filter_design}")
    
    def _memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _setup_combination_weights(self):
        """FIXED: Setup combination weights with consistent ordering"""
        
        # Define the EXACT order that scores are appended in forward()
        if self.filter == 'u':
            # Order: [direct, user_filtered]
            init_weights = torch.tensor([0.5, 0.5])
            self.score_order = ['direct', 'user']
            
        elif self.filter == 'i':
            # Order: [direct, item_filtered]
            init_weights = torch.tensor([0.5, 0.5])
            self.score_order = ['direct', 'item']
            
        elif self.filter == 'b':
            # Order: [direct, bipartite_filtered]
            init_weights = torch.tensor([0.5, 0.5])
            self.score_order = ['direct', 'bipartite']
            
        elif self.filter == 'ui':
            # Order: [direct, item_filtered, user_filtered]
            init_weights = torch.tensor([0.5, 0.3, 0.2])
            self.score_order = ['direct', 'item', 'user']
            
        elif self.filter == 'ub':
            # Order: [direct, user_filtered, bipartite_filtered]
            init_weights = torch.tensor([0.5, 0.3, 0.2])
            self.score_order = ['direct', 'user', 'bipartite']
            
        elif self.filter == 'uib':
            # Order: [direct, item_filtered, user_filtered, bipartite_filtered]
            init_weights = torch.tensor([0.4, 0.25, 0.25, 0.1])
            self.score_order = ['direct', 'item', 'user', 'bipartite']
            
        else:
            # Default fallback
            init_weights = torch.tensor([0.5, 0.5])
            self.score_order = ['direct', 'filtered']
        
        self.combination_weights = nn.Parameter(init_weights.to(self.device))
        
        print(f"   🎚️ Combination weights initialized: {init_weights.tolist()}")
        print(f"   📋 Score order: {self.score_order}")
    
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
        """FIXED forward pass with consistent scaling across all filter types"""
        if users.device != self.adj_tensor.device:
            users = users.to(self.adj_tensor.device)
        
        user_profiles = self.adj_tensor[users]  # [batch_size, n_items]
        user_filter_matrix, item_filter_matrix, bipartite_filter_matrix = self._get_filter_matrices()
        
        # CRITICAL FIX: Normalize user profiles for consistent baseline
        user_interaction_counts = user_profiles.sum(dim=1, keepdim=True) + 1e-8
        user_profiles_normalized = user_profiles / user_interaction_counts
        
        scores = [user_profiles]  # Direct collaborative filtering scores - ALWAYS FIRST
        
        # FIXED: Item-based filtering with consistent scaling (SECOND for i, ui, uib)
        if self.filter in ['i', 'ui', 'uib'] and item_filter_matrix is not None:
            # Use normalized profiles, then restore original magnitude
            item_filtered_normalized = user_profiles_normalized @ item_filter_matrix
            item_filtered = item_filtered_normalized * user_interaction_counts
            scores.append(item_filtered)
        
        # FIXED: User-based filtering with consistent scaling (SECOND for u, ub; THIRD for ui, uib)
        if self.filter in ['u', 'ui', 'uib', 'ub'] and user_filter_matrix is not None:
            # Apply user filter to normalized adjacency matrix
            adj_interaction_counts = self.adj_tensor.sum(dim=1, keepdim=True) + 1e-8
            adj_normalized = self.adj_tensor / adj_interaction_counts
            
            user_filtered_normalized = user_filter_matrix[users] @ adj_normalized
            # Scale to match user profile magnitude distribution
            user_filtered = user_filtered_normalized * user_interaction_counts
            scores.append(user_filtered)
        
        # FIXED: Bipartite filtering with consistent scaling (LAST for all combinations)
        if self.filter in ['b', 'uib', 'ub'] and bipartite_filter_matrix is not None:
            batch_size = users.shape[0]
            bipartite_input = torch.zeros(batch_size, self.n_users + self.n_items, device=self.device)
            
            # CRITICAL FIX: Use interaction density instead of one-hot
            user_density = user_profiles.sum(dim=1) / self.n_items  # Avg interactions per item
            bipartite_input[torch.arange(batch_size), users] = user_density
            
            bipartite_filtered = bipartite_input @ bipartite_filter_matrix
            bipartite_item_scores = bipartite_filtered[:, self.n_users:]
            
            # Scale to match user profile magnitude
            bipartite_scaled = bipartite_item_scores * user_interaction_counts
            scores.append(bipartite_scaled)
        
        # Verify correct number of scores
        expected_scores = len(self.combination_weights)
        actual_scores = len(scores)
        
        if expected_scores != actual_scores:
            raise RuntimeError(f"Score count mismatch! Expected {expected_scores}, got {actual_scores}")
        
        # Combine predictions using learnable weights
        weights = torch.softmax(self.combination_weights, dim=0)
        predicted = sum(w * score for w, score in zip(weights, scores))
        
        # Memory cleanup for large datasets
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
        if self.bipartite_filter is not None:
            filter_params.extend(self.bipartite_filter.parameters())
        return filter_params
    
    def get_other_parameters(self):
        """Get non-filter parameters"""
        filter_param_ids = {id(p) for p in self.get_filter_parameters()}
        return [p for p in self.parameters() if id(p) not in filter_param_ids]
    
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
    
    def debug_filter_learning(self):
        """Debug THREE-VIEW spectral filtering"""
        print(f"\n=== THREE-VIEW SPECTRAL FILTER DEBUG (FIXED) ===")
        print(f"🔍 THREE SPECTRAL VIEWS:")
        print(f"   1️⃣ User-User Similarity Matrix")
        print(f"   2️⃣ Item-Item Similarity Matrix") 
        print(f"   3️⃣ User-Item Bipartite Matrix")
        print(f"Similarity Type: {self.similarity_type}")
        print(f"Similarity Threshold: {self.similarity_threshold}")
        print(f"Filter Design: {self.filter_design}")
        print(f"Filter Type: {self.filter}")
        print(f"Device: {self.device}")
        print(f"User Eigenvalues: {self.u_n_eigen}")
        print(f"Item Eigenvalues: {self.i_n_eigen}")
        print(f"Bipartite Eigenvalues: {self.b_n_eigen}")
        
        with torch.no_grad():
            if self.filter in ['u', 'ui', 'uib', 'ub'] and self.user_filter is not None:
                print(f"\n👤 User Similarity Filter ({self.u_n_eigen} eigenvalues):")
                if hasattr(self, 'user_eigenvals'):
                    eigenvals = self.user_eigenvals.cpu().numpy()
                    print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
            
            if self.filter in ['i', 'ui', 'uib'] and self.item_filter is not None:
                print(f"\n🎬 Item Similarity Filter ({self.i_n_eigen} eigenvalues):")
                if hasattr(self, 'item_eigenvals'):
                    eigenvals = self.item_eigenvals.cpu().numpy()
                    print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
            
            if self.filter in ['b', 'uib', 'ub'] and self.bipartite_filter is not None:
                print(f"\n🔗 Bipartite Filter ({self.b_n_eigen} eigenvalues):")
                if hasattr(self, 'bipartite_eigenvals'):
                    eigenvals = self.bipartite_eigenvals.cpu().numpy()
                    print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
            
            weights = torch.softmax(self.combination_weights, dim=0)
            
            print(f"\n🎚️ THREE-VIEW Combination Weights:")
            for i, (name, weight) in enumerate(zip(self.score_order, weights.detach().cpu().numpy())):
                print(f"  {name}: {weight:.4f}")
        
        print("=== END THREE-VIEW DEBUG ===\n")
    
    def debug_score_magnitudes(self):
        """ADDED: Diagnostic method to verify scaling consistency"""
        print(f"\n=== SCORE MAGNITUDE ANALYSIS (FIXED) ===")
        
        # Sample a few users for testing
        test_users = torch.LongTensor([0, 1, 2, 3, 4]).to(self.device)
        user_profiles = self.adj_tensor[test_users]
        
        user_filter_matrix, item_filter_matrix, bipartite_filter_matrix = self._get_filter_matrices()
        
        print(f"Direct scores (user_profiles):")
        direct_stats = {
            'mean': user_profiles.mean().item(),
            'std': user_profiles.std().item(),
            'min': user_profiles.min().item(),
            'max': user_profiles.max().item()
        }
        print(f"  Mean: {direct_stats['mean']:.6f}, Std: {direct_stats['std']:.6f}")
        print(f"  Range: [{direct_stats['min']:.6f}, {direct_stats['max']:.6f}]")
        
        # Item filtering scores (FIXED)
        if self.filter in ['i', 'ui', 'uib'] and item_filter_matrix is not None:
            user_interaction_counts = user_profiles.sum(dim=1, keepdim=True) + 1e-8
            user_profiles_normalized = user_profiles / user_interaction_counts
            item_filtered_normalized = user_profiles_normalized @ item_filter_matrix
            item_scores = item_filtered_normalized * user_interaction_counts
            
            item_stats = {
                'mean': item_scores.mean().item(),
                'std': item_scores.std().item(),
                'min': item_scores.min().item(),
                'max': item_scores.max().item()
            }
            print(f"\nItem filtered scores (FIXED):")
            print(f"  Mean: {item_stats['mean']:.6f}, Std: {item_stats['std']:.6f}")
            print(f"  Range: [{item_stats['min']:.6f}, {item_stats['max']:.6f}]")
            print(f"  Magnitude ratio vs direct: {item_stats['mean']/direct_stats['mean']:.2f}x")
        
        # User filtering scores (FIXED)
        if self.filter in ['u', 'ui', 'uib', 'ub'] and user_filter_matrix is not None:
            adj_interaction_counts = self.adj_tensor.sum(dim=1, keepdim=True) + 1e-8
            adj_normalized = self.adj_tensor / adj_interaction_counts
            user_filtered_normalized = user_filter_matrix[test_users] @ adj_normalized
            user_interaction_counts = user_profiles.sum(dim=1, keepdim=True) + 1e-8
            user_scores = user_filtered_normalized * user_interaction_counts
            
            user_stats = {
                'mean': user_scores.mean().item(),
                'std': user_scores.std().item(),
                'min': user_scores.min().item(),
                'max': user_scores.max().item()
            }
            print(f"\nUser filtered scores (FIXED):")
            print(f"  Mean: {user_stats['mean']:.6f}, Std: {user_stats['std']:.6f}")
            print(f"  Range: [{user_stats['min']:.6f}, {user_stats['max']:.6f}]")
            print(f"  Magnitude ratio vs direct: {user_stats['mean']/direct_stats['mean']:.2f}x")
        
        # Bipartite filtering scores (FIXED)
        if self.filter in ['b', 'uib', 'ub'] and bipartite_filter_matrix is not None:
            batch_size = test_users.shape[0]
            bipartite_input = torch.zeros(batch_size, self.n_users + self.n_items, device=self.device)
            user_density = user_profiles.sum(dim=1) / self.n_items
            bipartite_input[torch.arange(batch_size), test_users] = user_density
            bipartite_filtered = bipartite_input @ bipartite_filter_matrix
            bipartite_item_scores = bipartite_filtered[:, self.n_users:]
            user_interaction_counts = user_profiles.sum(dim=1, keepdim=True) + 1e-8
            bipartite_scores = bipartite_item_scores * user_interaction_counts
            
            bipartite_stats = {
                'mean': bipartite_scores.mean().item(),
                'std': bipartite_scores.std().item(),
                'min': bipartite_scores.min().item(),
                'max': bipartite_scores.max().item()
            }
            print(f"\nBipartite filtered scores (FIXED):")
            print(f"  Mean: {bipartite_stats['mean']:.6f}, Std: {bipartite_stats['std']:.6f}")
            print(f"  Range: [{bipartite_stats['min']:.6f}, {bipartite_stats['max']:.6f}]")
            print(f"  Magnitude ratio vs direct: {bipartite_stats['mean']/direct_stats['mean']:.2f}x")
        
        # Show combination weights
        weights = torch.softmax(self.combination_weights, dim=0)
        print(f"\nCombination weights:")
        for name, weight in zip(self.score_order, weights):
            print(f"  {name}: {weight.item():.6f}")
        
        print(f"=== END MAGNITUDE ANALYSIS ===\n")
    
    def clear_cache(self):
        """Clear cache files for THREE-VIEW configuration"""
        cache_dir = "../cache"
        if not os.path.exists(cache_dir):
            return
        
        dataset = self.config.get('dataset', 'unknown')
        
        # Look for THREE_VIEW_FIXED cache files
        pattern_parts = [dataset, 'THREE_VIEW_FIXED']
        
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
# THREE-VIEW SPECTRAL FILTERING - FIXED VERSION
# ============================================================================
#
# 🔧 FIXES APPLIED:
#
# 1. **SCALING CONSISTENCY FIX**:
#    - All filter types now use normalized user profiles with magnitude restoration
#    - Item filter: normalize → filter → restore magnitude
#    - User filter: normalize adjacency → filter → scale to user magnitude  
#    - Bipartite filter: use density instead of one-hot → scale to user magnitude
#
# 2. **EIGENVALUE DOMAIN FIX**:
#    - Use similarity matrices consistently (eigenvalues 0-1, largest important)
#    - Apply 'LM' (largest magnitude) instead of 'SM' (smallest magnitude)
#    - Bipartite uses normalized adjacency instead of Laplacian
#
# 3. **COMBINATION WEIGHTS FIX**:
#    - Fixed ordering: always [direct, item, user, bipartite] when applicable
#    - Added score_order tracking for debugging
#    - Added verification of score count vs weight count
#
# 4. **CACHING FIX**:
#    - Updated cache keys to include "FIXED" to avoid conflicts
#    - Consistent cache key generation across all views
#
# 5. **DEBUGGING ENHANCEMENTS**:
#    - Added debug_score_magnitudes() method
#    - Enhanced debug_filter_learning() with eigenvalue ranges
#    - Better error reporting and verification
#
# 🎯 EXPECTED IMPROVEMENTS:
# - 'i' filter performance should significantly improve
# - All three filters should have similar score magnitudes
# - Three-view combinations should be more effective
# - More consistent training behavior
#
# ============================================================================