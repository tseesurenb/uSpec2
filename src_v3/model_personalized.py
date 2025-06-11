'''
Created on June 10, 2025
Enhanced Universal Spectral CF with THREE-VIEW Spectral Filtering and 2-HOP Support
UPDATED: Full support for 'ub' filter (User + Bipartite) with 2-hop propagation

This provides three complementary perspectives for collaborative filtering with multi-hop neighborhood exploration.

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


class OptimizedUserSpecificFilter(nn.Module):
    """OPTIMIZED user-specific filter - much faster than wrapper approach"""
    
    def __init__(self, n_users, filter_order=6, init_filter_name='smooth', 
                 shared_base=True, personalization_dim=8, filter_design='enhanced_basis'):
        super().__init__()
        self.n_users = n_users
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        self.shared_base = shared_base
        self.personalization_dim = personalization_dim
        self.filter_design = filter_design
        
        if shared_base:
            # APPROACH 1: Shared base filter + lightweight user adaptations
            self._setup_shared_base_filter()
        else:
            # APPROACH 2: Efficient user-specific parameters
            self._setup_fully_personalized_filter()
    
    def _setup_shared_base_filter(self):
        """Setup shared base filter with efficient user adaptations"""
        
        # Create base filter using the chosen design
        self.base_filter = self._create_base_filter()
        
        # Count base filter parameters
        base_param_count = sum(p.numel() for p in self.base_filter.parameters())
        print(f"      Base filter ({self.filter_design}) parameters: {base_param_count}")
        
        # User embeddings for personalization
        self.user_embeddings = nn.Embedding(self.n_users, self.personalization_dim)
        
        # OPTIMIZED: Direct parameter mapping instead of complex adaptation network
        if self.filter_design in ['original', 'chebyshev', 'legendre']:
            # For simple filters, use direct coefficient adaptation
            self.adaptation_layer = nn.Linear(self.personalization_dim, self.filter_order + 1)
            self.adaptation_type = 'coefficients'
        else:
            # For complex filters, use scaling factors
            self.adaptation_layer = nn.Linear(self.personalization_dim, base_param_count)
            self.adaptation_type = 'scaling'
        
        # Smaller adaptation scale for stability
        self.adaptation_scale = nn.Parameter(torch.tensor(0.1))
        
        # Initialize embeddings to small values
        nn.init.normal_(self.user_embeddings.weight, 0, 0.01)
        nn.init.xavier_uniform_(self.adaptation_layer.weight)
    
    def _setup_fully_personalized_filter(self):
        """Setup efficient fully personalized filters"""
        
        if self.filter_design in ['original', 'chebyshev', 'legendre']:
            # For simple filters, store coefficients directly
            base_coeffs = fl.get_filter_coefficients(self.init_filter_name, order=self.filter_order, as_tensor=True)
            if len(base_coeffs) < self.filter_order + 1:
                padded_coeffs = torch.zeros(self.filter_order + 1)
                padded_coeffs[:len(base_coeffs)] = base_coeffs
                base_coeffs = padded_coeffs
            elif len(base_coeffs) > self.filter_order + 1:
                base_coeffs = base_coeffs[:self.filter_order + 1]
            
            # Initialize with small random variations around base
            self.user_coeffs = nn.Parameter(
                base_coeffs.unsqueeze(0).repeat(self.n_users, 1) + 
                torch.randn(self.n_users, self.filter_order + 1) * 0.02
            )
            self.param_type = 'coefficients'
        else:
            # For complex filters, use shared base with user scaling
            print(f"⚠️ Complex filter '{self.filter_design}' using shared base approach for efficiency")
            self._setup_shared_base_filter()
    
    def _create_base_filter(self):
        """Create base filter of chosen design"""
        if self.filter_design == 'original':
            return fl.UniversalSpectralFilter(self.filter_order, self.init_filter_name)
        elif self.filter_design == 'basis':
            return fl.SpectralBasisFilter(self.filter_order, self.init_filter_name)
        elif self.filter_design == 'enhanced_basis':
            return fl.EnhancedSpectralBasisFilter(self.filter_order, self.init_filter_name)
        elif self.filter_design == 'adaptive_golden':
            return fl.AdaptiveGoldenFilter(self.filter_order, self.init_filter_name)
        elif self.filter_design == 'multiscale':
            return fl.MultiScaleSpectralFilter(self.filter_order, self.init_filter_name)
        elif self.filter_design == 'ensemble':
            return fl.EnsembleSpectralFilter(self.filter_order, self.init_filter_name)
        elif self.filter_design == 'chebyshev':
            return fl.ChebyshevSpectralFilter(self.filter_order, self.init_filter_name)
        elif self.filter_design == 'legendre':
            return fl.LegendreSpectralFilter(self.filter_order, self.init_filter_name)
        else:
            print(f"⚠️ Filter design '{self.filter_design}' not fully supported, using enhanced_basis")
            return fl.EnhancedSpectralBasisFilter(self.filter_order, self.init_filter_name)
    
    def forward(self, eigenvalues, user_ids):
        """OPTIMIZED forward pass - much faster"""
        device = eigenvalues.device
        batch_size = user_ids.shape[0]
        
        if not self.shared_base and hasattr(self, 'user_coeffs'):
            # FAST PATH: Direct coefficient lookup for simple filters
            user_coeffs = self.user_coeffs[user_ids].to(device)  # [batch_size, filter_order + 1]
            return self._apply_polynomial_filter(eigenvalues, user_coeffs)
        
        elif self.shared_base:
            # OPTIMIZED PATH: Shared base with efficient adaptations
            if self.adaptation_type == 'coefficients':
                # For simple filters: adapt coefficients directly
                user_embeds = self.user_embeddings(user_ids)  # [batch_size, personalization_dim]
                adaptations = self.adaptation_layer(user_embeds)  # [batch_size, filter_order + 1]
                adaptations = torch.tanh(adaptations) * self.adaptation_scale  # Bound adaptations
                
                # Get base coefficients (for simple filters)
                if hasattr(self.base_filter, 'coeffs'):
                    base_coeffs = self.base_filter.coeffs.unsqueeze(0)  # [1, filter_order + 1]
                else:
                    # Fallback: use smooth coefficients
                    base_coeffs = fl.get_filter_coefficients(self.init_filter_name, order=self.filter_order, as_tensor=True)
                    base_coeffs = base_coeffs.to(device).unsqueeze(0)
                
                user_coeffs = base_coeffs + adaptations  # [batch_size, filter_order + 1]
                return self._apply_polynomial_filter(eigenvalues, user_coeffs)
            
            else:
                # For complex filters: use base filter with scaling
                base_response = self.base_filter(eigenvalues)  # [n_eigenvals]
                
                user_embeds = self.user_embeddings(user_ids)  # [batch_size, personalization_dim]
                scaling_factors = self.adaptation_layer(user_embeds)  # [batch_size, n_params]
                
                # Simple approach: use first scaling factor for response scaling
                user_scales = torch.sigmoid(scaling_factors[:, 0:1]) + 0.5  # [batch_size, 1], range [0.5, 1.5]
                
                # Apply user-specific scaling
                user_responses = base_response.unsqueeze(0) * user_scales  # [batch_size, n_eigenvals]
                return user_responses
        
        else:
            # Fallback: use base filter
            base_response = self.base_filter(eigenvalues)
            return base_response.unsqueeze(0).repeat(batch_size, 1)
    
    def _apply_polynomial_filter(self, eigenvalues, user_coeffs):
        """Apply Chebyshev polynomial filter efficiently"""
        device = eigenvalues.device
        batch_size, n_coeffs = user_coeffs.shape
        
        # Normalize eigenvalues to [-1, 1]
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1  # [n_eigenvals]
        
        # Efficient Chebyshev computation using recurrence
        if n_coeffs >= 1:
            result = user_coeffs[:, 0:1]  # [batch_size, 1]
            
        if n_coeffs >= 2:
            result = result + user_coeffs[:, 1:2] * x.unsqueeze(0)  # [batch_size, n_eigenvals]
            
        if n_coeffs >= 3:
            T_prev = torch.ones_like(x).unsqueeze(0)  # [1, n_eigenvals]
            T_curr = x.unsqueeze(0)  # [1, n_eigenvals]
            
            for i in range(2, n_coeffs):
                T_next = 2 * x.unsqueeze(0) * T_curr - T_prev
                result = result + user_coeffs[:, i:i+1] * T_next
                T_prev, T_curr = T_curr, T_next
        
        # Apply activation
        return torch.exp(-torch.abs(result).clamp(max=10.0)) + 1e-6


class UserSpecificUniversalSpectralCF(nn.Module):
    """COMPLETE User-Specific Universal Spectral CF with 2-HOP SUPPORT"""
    
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.filter_order = self.config.get('filter_order', 6)
        self.filter = self.config.get('filter', 'ui')
        
        # 2-HOP Configuration
        self.n_hops = self.config.get('n_hops', 1)  # 1 or 2 hops
        self.hop_decay = self.config.get('hop_decay', 0.5)  # Decay factor for 2nd hop
        
        # User-specific configuration
        self.shared_base = self.config.get('shared_base', True)
        self.personalization_dim = self.config.get('personalization_dim', 8)
        self.cold_start_strategy = self.config.get('cold_start_strategy', 'average')
        
        # Filter design configuration
        self.filter_design = self.config.get('filter_design', 'enhanced_basis')
        self.init_filter = self.config.get('init_filter', 'smooth')
        
        # OPTIMIZATION: Reduce eigenvalue counts for faster training
        self.enable_fast_mode = self.config.get('fast_mode', True)
        
        # Convert and get dataset characteristics
        adj_dense = adj_mat.toarray() if sp.issparse(adj_mat) else adj_mat
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32).to(self.device))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Calculate dataset characteristics
        total_interactions = torch.sum(self.adj_tensor).item()
        sparsity = total_interactions / (self.n_users * self.n_items)
        
        # ENHANCED: Calculate eigenvalues for all THREE views
        self.u_n_eigen, self.i_n_eigen, self.b_n_eigen = self._calculate_optimized_eigenvalues(total_interactions, sparsity)
        
        # Similarity parameters
        self.similarity_type = self.config.get('similarity_type', 'cosine')
        self.similarity_threshold = self.config.get('similarity_threshold', 0.01)
        
        print(f"🚀 2-HOP USER-SPECIFIC Universal Spectral CF:")
        print(f"   └─ Dataset: {self.config.get('dataset', 'unknown')}")
        print(f"   └─ Users: {self.n_users:,}, Items: {self.n_items:,}")
        print(f"   └─ Interactions: {int(total_interactions):,}")
        print(f"   🎯 2-HOP CONFIGURATION:")
        print(f"      Hops: {self.n_hops} ({'2-hop' if self.n_hops == 2 else '1-hop'})")
        print(f"      Hop decay: {self.hop_decay}")
        print(f"   🔍 THREE SPECTRAL VIEWS:")
        print(f"      1️⃣ User-User: {self.u_n_eigen} eigenvalues")
        print(f"      2️⃣ Item-Item: {self.i_n_eigen} eigenvalues")
        print(f"      3️⃣ User-Item Bipartite: {self.b_n_eigen} eigenvalues")
        print(f"   🎯 USER-SPECIFIC SETTINGS:")
        print(f"      Fast Mode: {'Enabled' if self.enable_fast_mode else 'Disabled'}")
        print(f"      Shared Base: {'Yes' if self.shared_base else 'No'}")
        print(f"      Personalization Dim: {self.personalization_dim}")
        print(f"   🎨 FILTER CONFIGURATION:")
        print(f"      Filter Design: {self.filter_design}")
        print(f"      Init Filter: {self.init_filter}")
        print(f"      Filter Type: {self.filter}")
        print(f"   └─ Device: {self.device}")
        
        # Setup optimized spectral filtering with BIPARTITE support
        self._setup_optimized_filters()
        self._setup_combination_weights()
    
    def _calculate_optimized_eigenvalues(self, total_interactions, sparsity):
        """Calculate optimized eigenvalues for ALL THREE VIEWS"""
        manual_u_eigen = self.config.get('u_n_eigen', None)
        manual_i_eigen = self.config.get('i_n_eigen', None)
        manual_b_eigen = self.config.get('b_n_eigen', None)  # BIPARTITE
        manual_n_eigen = self.config.get('n_eigen', None)
        
        if manual_u_eigen is not None and manual_i_eigen is not None:
            if manual_u_eigen > 0 and manual_i_eigen > 0:
                # Handle bipartite eigenvalues
                if manual_b_eigen is not None and manual_b_eigen > 0:
                    print(f"   🎯 Using manual eigenvalues: u={manual_u_eigen}, i={manual_i_eigen}, b={manual_b_eigen}")
                    return manual_u_eigen, manual_i_eigen, manual_b_eigen
                else:
                    # Auto-calculate bipartite eigenvalues
                    b_n_eigen = self._calculate_bipartite_eigenvalues(total_interactions, sparsity)
                    print(f"   🎯 Using manual u={manual_u_eigen}, i={manual_i_eigen}, auto b={b_n_eigen}")
                    return manual_u_eigen, manual_i_eigen, b_n_eigen
        elif manual_n_eigen is not None and manual_n_eigen > 0:
            print(f"   🎯 Using manual n_eigen for all views: {manual_n_eigen}")
            return manual_n_eigen, manual_n_eigen, manual_n_eigen
        
        # OPTIMIZED: Smaller eigenvalue counts for faster training
        if self.enable_fast_mode:
            # Aggressive reduction for speed
            base_u = min(32, max(16, self.n_users // 50))
            base_i = min(48, max(24, self.n_items // 50))
            base_b = min(40, max(20, (self.n_users + self.n_items) // 80))
        else:
            # Standard calculation
            if self.n_users < 1000:
                base_u = 32
            elif self.n_users < 5000:
                base_u = 48
            else:
                base_u = 64
                
            if self.n_items < 1000:
                base_i = 32
            elif self.n_items < 5000:
                base_i = 48
            else:
                base_i = 64
            
            # Bipartite eigenvalues
            base_b = self._calculate_bipartite_eigenvalues(total_interactions, sparsity)
        
        print(f"   🚀 Optimized eigenvalues: u={base_u}, i={base_i}, b={base_b} ({'fast mode' if self.enable_fast_mode else 'standard'})")
        return base_u, base_i, base_b
    
    def _calculate_bipartite_eigenvalues(self, total_interactions, sparsity):
        """Calculate eigenvalues for bipartite graph"""
        bipartite_size = self.n_users + self.n_items
        
        if bipartite_size < 1000:
            base_eigen = 24
        elif bipartite_size < 5000:
            base_eigen = 48
        elif bipartite_size < 10000:
            base_eigen = 64
        else:
            base_eigen = 80
        
        # Adjust for sparsity and interactions
        avg_degree = (2 * total_interactions) / bipartite_size
        
        if avg_degree < 5:
            base_eigen = int(base_eigen * 1.2)  # Need more eigenvalues for sparse graphs
        elif avg_degree > 50:
            base_eigen = int(base_eigen * 0.8)  # Can use fewer for dense graphs
        
        if sparsity < 0.001:
            base_eigen = int(base_eigen * 1.3)
        elif sparsity > 0.05:
            base_eigen = int(base_eigen * 0.7)
        
        return max(16, min(base_eigen, 128))
    
    def _get_cache_path(self, cache_type, filter_type=None):
        """Generate cache file path with THREE-VIEW parameters"""
        cache_dir = "../cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = self.config.get('dataset', 'unknown')
        sim_type = self.similarity_type
        threshold = str(self.similarity_threshold).replace('.', 'p')
        u_eigen = self.u_n_eigen
        i_eigen = self.i_n_eigen
        b_eigen = self.b_n_eigen
        filter_design = self.filter_design
        filter_mode = self.filter
        fast_mode = 'fast' if self.enable_fast_mode else 'normal'
        shared_mode = 'shared' if self.shared_base else 'personal'
        hop_mode = f'{self.n_hops}hop'
        
        base_name = f"{dataset}_USER_SPECIFIC_2HOP_{sim_type}_th{threshold}_u{u_eigen}_i{i_eigen}_b{b_eigen}_{filter_design}_{filter_mode}_{fast_mode}_{shared_mode}_{hop_mode}"
        
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
        similarity_thresholded = torch.where(similarity >= self.similarity_threshold, 
                                           similarity, torch.zeros_like(similarity))
        similarity_thresholded.fill_diagonal_(1.0)
        result = torch.clamp(similarity_thresholded, min=0.0, max=1.0)
        
        if cache_type:
            self._save_to_cache(result.cpu(), cache_path)
        
        return result
    
    def _compute_similarity_laplacian(self, similarity_matrix):
        """Compute normalized similarity Laplacian"""
        degree = similarity_matrix.sum(dim=1) + 1e-8
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        normalized_laplacian = similarity_matrix * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
        identity = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
        laplacian = identity - normalized_laplacian
        
        return laplacian
    
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
        
        # Bipartite Laplacian: L = I - normalized_adjacency
        identity = torch.eye(n_total, device=self.device)
        bipartite_laplacian = identity - normalized_bipartite
        
        print(f"    Bipartite Laplacian shape: {bipartite_laplacian.shape}")
        
        return bipartite_laplacian
    
    def _setup_optimized_filters(self):
        """Setup optimized spectral filters with FULL THREE-VIEW SUPPORT"""
        print(f"Computing THREE-VIEW eigendecompositions for filter type: {self.filter}")
        start = time.time()
        
        self.user_filter = None
        self.item_filter = None
        self.bipartite_filter = None  # BIPARTITE SUPPORT
        
        if self.filter in ['u', 'ui', 'uib', 'ub']:  # UPDATED: Added 'ub'
            print("1️⃣ Processing user-user similarity matrix...")
            self.user_filter = self._create_optimized_filter('user')
            self._memory_cleanup()
        
        if self.filter in ['i', 'ui', 'uib']:  # NOTE: 'ub' does NOT include item filter
            print("2️⃣ Processing item-item similarity matrix...")
            self.item_filter = self._create_optimized_filter('item')
            self._memory_cleanup()
        
        # NEW: BIPARTITE FILTER SUPPORT
        if self.filter in ['b', 'uib', 'ub']:  # UPDATED: Added 'ub'
            print("3️⃣ Processing user-item bipartite matrix...")
            self.bipartite_filter = self._create_bipartite_filter()
            self._memory_cleanup()
        
        print(f'THREE-VIEW optimized filter setup completed in {time.time() - start:.2f}s')
    
    def _create_optimized_filter(self, filter_type):
        """Create optimized filter with eigendecomposition"""
        
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
                
                self._save_to_cache((eigenvals_tensor, eigenvecs_tensor), eigen_cache_path)
                
                self.register_buffer(f'{filter_type}_eigenvals', eigenvals_tensor.to(self.device))
                self.register_buffer(f'{filter_type}_eigenvecs', eigenvecs_tensor.to(self.device))
                
                print(f"  {filter_type.capitalize()} eigendecomposition: {k} components")
                
            except Exception as e:
                print(f"  {filter_type.capitalize()} eigendecomposition failed: {e}")
                eigenvals = np.linspace(0, 1, min(n_eigen_to_use, n_components))
                eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
                eigenvecs_tensor = torch.eye(n_components, min(n_eigen_to_use, n_components))
                
                self.register_buffer(f'{filter_type}_eigenvals', eigenvals_tensor.to(self.device))
                self.register_buffer(f'{filter_type}_eigenvecs', eigenvecs_tensor.to(self.device))
            
            del laplacian_np
            if 'eigenvals' in locals():
                del eigenvals, eigenvecs
            self._memory_cleanup()
        
        # Create optimized user-specific filter
        print(f"  Creating optimized {self.filter_design} filter for {filter_type}...")
        return OptimizedUserSpecificFilter(
            n_users=self.n_users,
            filter_order=self.filter_order,
            init_filter_name=self.init_filter,
            shared_base=self.shared_base,
            personalization_dim=self.personalization_dim,
            filter_design=self.filter_design
        )
    
    def _create_bipartite_filter(self):
        """NEW: Create bipartite filter for personalized model"""
        
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
        
        # Return optimized user-specific filter for bipartite
        print(f"  Creating optimized {self.filter_design} bipartite filter...")
        return OptimizedUserSpecificFilter(
            n_users=self.n_users,
            filter_order=self.filter_order,
            init_filter_name=self.init_filter,
            shared_base=self.shared_base,
            personalization_dim=self.personalization_dim,
            filter_design=self.filter_design
        )
    
    def _memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _setup_combination_weights(self):
        """Setup learnable combination weights for THREE VIEWS and 2-HOP"""
        base_components = []
        
        if self.filter == 'u':
            base_components = ['direct', 'user_filtered']
        elif self.filter == 'i':
            base_components = ['direct', 'item_filtered']
        elif self.filter == 'b':
            base_components = ['direct', 'bipartite_filtered']
        elif self.filter == 'ui':
            base_components = ['direct', 'item_filtered', 'user_filtered']
        elif self.filter == 'ub':
            base_components = ['direct', 'user_filtered', 'bipartite_filtered']
        elif self.filter == 'uib':
            base_components = ['direct', 'item_filtered', 'user_filtered', 'bipartite_filtered']
        else:
            base_components = ['direct', 'filtered']
        
        # For 2-hop, double the number of components (1-hop + 2-hop)
        if self.n_hops == 2:
            hop_components = []
            for comp in base_components:
                hop_components.extend([f"{comp}_1hop", f"{comp}_2hop"])
            n_components = len(hop_components)
            
            # Initialize with decay for 2-hop components
            init_weights = torch.ones(n_components)
            for i in range(len(base_components)):
                init_weights[2*i] = 1.0  # 1-hop weight
                init_weights[2*i+1] = self.hop_decay  # 2-hop weight (decayed)
            
            print(f"   🎚️  2-HOP combination weights: {hop_components}")
        else:
            # Standard 1-hop weights
            if self.filter == 'u':
                init_weights = torch.tensor([0.5, 0.5])
            elif self.filter == 'i':
                init_weights = torch.tensor([0.5, 0.5])
            elif self.filter == 'b':
                init_weights = torch.tensor([0.5, 0.5])
            elif self.filter == 'ui':
                init_weights = torch.tensor([0.5, 0.3, 0.2])
            elif self.filter == 'ub':
                init_weights = torch.tensor([0.5, 0.3, 0.2])
            elif self.filter == 'uib':
                init_weights = torch.tensor([0.4, 0.25, 0.25, 0.1])
            else:
                init_weights = torch.tensor([0.5, 0.5])
            
            print(f"   🎚️  1-HOP combination weights: {base_components}")
        
        self.combination_weights = nn.Parameter(init_weights.to(self.device))
        print(f"   🎯 Initialized with decay factor: {self.hop_decay}")
    
    def forward(self, users):
        """2-HOP forward pass with personalized spectral filtering"""
        if users.device != self.adj_tensor.device:
            users = users.to(self.adj_tensor.device)
        
        batch_size = users.shape[0]
        user_profiles = self.adj_tensor[users]  # [batch_size, n_items]
        
        # Collect scores from both 1-hop and 2-hop
        all_scores = []
        
        # 1-HOP SCORES
        hop1_scores = self._compute_1hop_scores(users, user_profiles)
        all_scores.extend(hop1_scores)
        
        # 2-HOP SCORES (if enabled)
        if self.n_hops == 2:
            hop2_scores = self._compute_2hop_scores(users, user_profiles)
            all_scores.extend(hop2_scores)
        
        # Combine all scores using learnable weights
        weights = torch.softmax(self.combination_weights, dim=0)
        predicted = sum(w * score for w, score in zip(weights, all_scores))
        
        return predicted
    
    def _compute_1hop_scores(self, users, user_profiles):
        """Compute 1-hop scores with FULL USER PERSONALIZATION (corrected)"""
        scores = [user_profiles]  # Direct collaborative filtering scores
        
        # Item-based filtering (1-hop) - PERSONALIZED PER USER
        if self.filter in ['i', 'ui', 'uib'] and self.item_filter is not None:
            # Get PERSONALIZED item filter responses for each user
            item_responses = self.item_filter(self.item_eigenvals, users)  # [batch_size, n_eigenvals]
            
            # Apply personalized filtering for each user individually
            item_filtered_batch = []
            for i, user_id in enumerate(users):
                # Get this user's personalized item matrix
                user_item_response = item_responses[i]  # [n_eigenvals]
                user_item_matrix = self.item_eigenvecs @ torch.diag(user_item_response) @ self.item_eigenvecs.t()
                
                # Apply this user's personalized item filter
                user_item_filtered = user_profiles[i:i+1] @ user_item_matrix  # [1, n_items]
                item_filtered_batch.append(user_item_filtered)
            
            item_filtered = torch.cat(item_filtered_batch, dim=0)  # [batch_size, n_items]
            scores.append(item_filtered)
        
        # User-based filtering (1-hop) - PERSONALIZED PER USER
        if self.filter in ['u', 'ui', 'uib', 'ub'] and self.user_filter is not None:
            # Get PERSONALIZED user filter responses for each user
            user_responses = self.user_filter(self.user_eigenvals, users)  # [batch_size, n_eigenvals]
            
            # Apply personalized filtering for each user individually
            user_filtered_batch = []
            for i, user_id in enumerate(users):
                # Get this user's personalized user matrix
                user_user_response = user_responses[i]  # [n_eigenvals]
                user_user_matrix = self.user_eigenvecs @ torch.diag(user_user_response) @ self.user_eigenvecs.t()
                
                # Apply this user's personalized user filter: User → Similar Users → Items
                user_filtered_scores = user_user_matrix[user_id:user_id+1] @ self.adj_tensor  # [1, n_items]
                user_filtered_batch.append(user_filtered_scores)
            
            user_filtered = torch.cat(user_filtered_batch, dim=0)  # [batch_size, n_items]
            scores.append(user_filtered)
        
        # Bipartite filtering (1-hop) - PERSONALIZED PER USER
        if self.filter in ['b', 'uib', 'ub'] and self.bipartite_filter is not None:
            # Get PERSONALIZED bipartite filter responses for each user
            bipartite_responses = self.bipartite_filter(self.bipartite_eigenvals, users)  # [batch_size, n_eigenvals]
            
            # Apply personalized filtering for each user individually
            bipartite_filtered_batch = []
            for i, user_id in enumerate(users):
                # Get this user's personalized bipartite matrix
                user_bipartite_response = bipartite_responses[i]  # [n_eigenvals]
                user_bipartite_matrix = self.bipartite_eigenvecs @ torch.diag(user_bipartite_response) @ self.bipartite_eigenvecs.t()
                
                # Extract item-item part for this user's personalized filter
                user_item_part = user_bipartite_matrix[self.n_users:, self.n_users:]  # [n_items, n_items]
                
                # Apply this user's personalized bipartite filter
                user_bipartite_scores = user_profiles[i:i+1] @ user_item_part  # [1, n_items]
                bipartite_filtered_batch.append(user_bipartite_scores)
            
            bipartite_filtered = torch.cat(bipartite_filtered_batch, dim=0)  # [batch_size, n_items]
            scores.append(bipartite_filtered)
        
        return scores
    
    def _compute_2hop_scores(self, users, user_profiles):
        """Compute 2-hop scores with PERSONALIZED propagation"""
        scores = []
        
        # 2-hop direct: User → Users → Items → Items (non-personalized baseline)
        direct_2hop = self._propagate_through_users_items(user_profiles)
        scores.append(direct_2hop)
        
        # Item-based 2-hop: User → Items → Items → Items (PERSONALIZED)
        if self.filter in ['i', 'ui', 'uib'] and self.item_filter is not None:
            # Get PERSONALIZED item filter responses for each user
            item_responses = self.item_filter(self.item_eigenvals, users)  # [batch_size, n_eigenvals]
            
            # Apply personalized 2-hop propagation for each user
            item_2hop_batch = []
            for i, user_id in enumerate(users):
                # Get this user's personalized item matrix
                user_item_response = item_responses[i]  # [n_eigenvals]
                user_item_matrix = self.item_eigenvecs @ torch.diag(user_item_response) @ self.item_eigenvecs.t()
                
                # Apply 2-hop for this specific user: Items → Items → Items
                user_item_2hop = user_profiles[i:i+1] @ user_item_matrix @ user_item_matrix
                item_2hop_batch.append(user_item_2hop)
            
            item_2hop = torch.cat(item_2hop_batch, dim=0)  # [batch_size, n_items]
            scores.append(item_2hop)
        
        # User-based 2-hop: User → Users → Users → Items (PERSONALIZED)
        if self.filter in ['u', 'ui', 'uib', 'ub'] and self.user_filter is not None:
            # Get PERSONALIZED user filter responses for each user
            user_responses = self.user_filter(self.user_eigenvals, users)  # [batch_size, n_eigenvals]
            
            # Apply personalized 2-hop propagation for each user
            user_2hop_batch = []
            for i, user_id in enumerate(users):
                # Get this user's personalized user matrix
                user_user_response = user_responses[i]  # [n_eigenvals]
                user_user_matrix = self.user_eigenvecs @ torch.diag(user_user_response) @ self.user_eigenvecs.t()
                
                # Apply 2-hop for this specific user: User → Users → Users → Items
                # Start with one-hot user representation
                user_repr = torch.zeros(1, self.n_users, device=self.device)
                user_repr[0, user_id] = 1.0
                
                # 2-hop propagation through personalized user similarities
                user_2hop_repr = user_repr @ user_user_matrix @ user_user_matrix
                user_2hop_scores = user_2hop_repr @ self.adj_tensor  # [1, n_items]
                user_2hop_batch.append(user_2hop_scores)
            
            user_2hop = torch.cat(user_2hop_batch, dim=0)  # [batch_size, n_items]
            scores.append(user_2hop)
        
        # Bipartite 2-hop: User ↔ Items ↔ Users ↔ Items (PERSONALIZED)
        if self.filter in ['b', 'uib', 'ub'] and self.bipartite_filter is not None:
            # Get PERSONALIZED bipartite filter responses for each user
            bipartite_responses = self.bipartite_filter(self.bipartite_eigenvals, users)  # [batch_size, n_eigenvals]
            
            # Apply personalized 2-hop propagation for each user
            bipartite_2hop_batch = []
            for i, user_id in enumerate(users):
                # Get this user's personalized bipartite matrix
                user_bipartite_response = bipartite_responses[i]  # [n_eigenvals]
                user_bipartite_matrix = self.bipartite_eigenvecs @ torch.diag(user_bipartite_response) @ self.bipartite_eigenvecs.t()
                
                # Apply 2-hop bipartite propagation for this user
                bipartite_input = torch.zeros(1, self.n_users + self.n_items, device=self.device)
                bipartite_input[0, user_id] = 1.0
                
                # 2-hop: bipartite_matrix^2 (personalized)
                user_bipartite_2hop_matrix = user_bipartite_matrix @ user_bipartite_matrix
                bipartite_2hop_filtered = bipartite_input @ user_bipartite_2hop_matrix
                bipartite_2hop_scores = bipartite_2hop_filtered[:, self.n_users:]  # [1, n_items]
                bipartite_2hop_batch.append(bipartite_2hop_scores)
            
            bipartite_2hop = torch.cat(bipartite_2hop_batch, dim=0)  # [batch_size, n_items]
            scores.append(bipartite_2hop)
        
        return scores
    
    def _propagate_through_users_items(self, user_profiles):
        """2-hop propagation: User → Items → Users → Items"""
        # user_profiles: [batch_size, n_items]
        # adj_tensor: [n_users, n_items]
        
        # Step 1: User → Items (already have user_profiles)
        # Step 2: Items → Users (find which users are connected to these items)
        # For each item, find connected users: adj_tensor.t() @ ones gives user connections
        # We want: for each batch user's items, find other users who also like those items
        
        # Method: user_profiles @ adj_tensor.t() gives [batch_size, n_users]
        # This shows how each batch user relates to all users through shared items
        user_similarities_via_items = user_profiles @ self.adj_tensor.t()  # [batch_size, n_users]
        
        # Step 3: Users → Items (propagate through user similarities to get final items)
        # user_similarities_via_items @ adj_tensor gives [batch_size, n_items]
        users_to_items_2hop = user_similarities_via_items @ self.adj_tensor  # [batch_size, n_items]
        
        return users_to_items_2hop
    
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
        
        user_specific_params = 0
        if hasattr(self, 'user_filter') and self.user_filter is not None:
            if hasattr(self.user_filter, 'user_embeddings'):
                user_specific_params += self.user_filter.user_embeddings.weight.numel()
            if hasattr(self.user_filter, 'adaptation_layer'):
                user_specific_params += sum(p.numel() for p in self.user_filter.adaptation_layer.parameters())
            if hasattr(self.user_filter, 'user_coeffs'):
                user_specific_params += self.user_filter.user_coeffs.numel()
        
        # Add bipartite-specific parameters
        bipartite_params = 0
        if hasattr(self, 'bipartite_filter') and self.bipartite_filter is not None:
            if hasattr(self.bipartite_filter, 'user_embeddings'):
                bipartite_params += self.bipartite_filter.user_embeddings.weight.numel()
            if hasattr(self.bipartite_filter, 'adaptation_layer'):
                bipartite_params += sum(p.numel() for p in self.bipartite_filter.adaptation_layer.parameters())
            if hasattr(self.bipartite_filter, 'user_coeffs'):
                bipartite_params += self.bipartite_filter.user_coeffs.numel()
        
        return {
            'total': total_params,
            'filter': filter_params,
            'user_specific': user_specific_params,
            'bipartite_specific': bipartite_params,
            'combination': self.combination_weights.numel(),
            'other': total_params - filter_params
        }
    
    def debug_filter_learning(self):
        """Debug 2-HOP user-specific spectral filtering"""
        print(f"\n=== 2-HOP USER-SPECIFIC FILTER DEBUG ===")
        print(f"Filter Design: {self.filter_design}")
        print(f"Hops: {self.n_hops} ({'2-hop enabled' if self.n_hops == 2 else '1-hop only'})")
        print(f"Hop decay: {self.hop_decay}")
        print(f"Fast Mode: {'Enabled' if self.enable_fast_mode else 'Disabled'}")
        print(f"Shared Base: {self.shared_base}")
        print(f"User Eigenvalues: {self.u_n_eigen}")
        print(f"Item Eigenvalues: {self.i_n_eigen}")
        print(f"Bipartite Eigenvalues: {self.b_n_eigen}")
        print(f"Filter Type: {self.filter}")
        
        param_count = self.get_parameter_count()
        print(f"\nParameter Count:")
        print(f"  Total: {param_count['total']:,}")
        print(f"  User-specific: {param_count['user_specific']:,}")
        print(f"  Bipartite-specific: {param_count['bipartite_specific']:,}")
        print(f"  Filter: {param_count['filter']:,}")
        
        # Show combination weights with hop information
        weights = torch.softmax(self.combination_weights, dim=0)
        print(f"\n🎚️  2-HOP Combination Weights:")
        
        if self.n_hops == 2:
            weight_names = []
            if self.filter == 'uib':
                weight_names = ['Direct_1hop', 'Direct_2hop', 'Item_1hop', 'Item_2hop', 
                              'User_1hop', 'User_2hop', 'Bipartite_1hop', 'Bipartite_2hop']
            elif self.filter == 'ui':
                weight_names = ['Direct_1hop', 'Direct_2hop', 'Item_1hop', 'Item_2hop', 
                              'User_1hop', 'User_2hop']
            elif self.filter == 'ub':
                weight_names = ['Direct_1hop', 'Direct_2hop', 'User_1hop', 'User_2hop', 
                              'Bipartite_1hop', 'Bipartite_2hop']
            elif self.filter == 'b':
                weight_names = ['Direct_1hop', 'Direct_2hop', 'Bipartite_1hop', 'Bipartite_2hop']
            else:
                weight_names = ['Direct_1hop', 'Direct_2hop', 'Filtered_1hop', 'Filtered_2hop']
        else:
            if self.filter == 'uib':
                weight_names = ['Direct', 'Item', 'User', 'Bipartite']
            elif self.filter == 'ui':
                weight_names = ['Direct', 'Item', 'User']
            elif self.filter == 'ub':
                weight_names = ['Direct', 'User', 'Bipartite']
            elif self.filter == 'b':
                weight_names = ['Direct', 'Bipartite']
            else:
                weight_names = ['Direct', 'Filtered']
        
        for i, (name, weight) in enumerate(zip(weight_names, weights.detach().cpu().numpy())):
            hop_info = "🔄" if "2hop" in name else "🎯" if "1hop" in name else "📊"
            print(f"  {hop_info} {name}: {weight:.4f}")
        
        # Analyze hop contribution
        if self.n_hops == 2:
            hop1_total = sum(weights[i] for i in range(0, len(weights), 2))
            hop2_total = sum(weights[i] for i in range(1, len(weights), 2))
            print(f"\n🔍 Hop Analysis:")
            print(f"  1-hop total contribution: {hop1_total:.4f}")
            print(f"  2-hop total contribution: {hop2_total:.4f}")
            print(f"  2-hop/1-hop ratio: {hop2_total/hop1_total:.3f}")
        
        print("=== END 2-HOP DEBUG ===\n")
    
    def clear_cache(self):
        """Clear cache files"""
        cache_dir = "../cache"
        if not os.path.exists(cache_dir):
            return
        
        dataset = self.config.get('dataset', 'unknown')
        pattern_parts = [dataset, 'USER_SPECIFIC_2HOP']
        
        removed_count = 0
        for filename in os.listdir(cache_dir):
            if all(part in filename for part in pattern_parts):
                file_path = os.path.join(cache_dir, filename)
                try:
                    os.remove(file_path)
                    removed_count += 1
                except:
                    pass
        
        if removed_count > 0:
            print(f"Removed {removed_count} user-specific 2-hop cache files")


# ============================================================================
# 2-HOP USER-SPECIFIC SPECTRAL FILTERING - SUMMARY
# ============================================================================
#
# 🚀 NEW 2-HOP CAPABILITIES:
#
# 1. **Dual-Hop Architecture**:
#    - 1-hop: Direct neighbors (User → Similar users → Items)
#    - 2-hop: Extended neighbors (User → Similar users → Similar users → Items)
#
# 2. **Multi-hop Spectral Filtering**:
#    - Each hop uses personalized spectral filters
#    - Independent filter responses for each user at each hop
#    - Learnable combination of 1-hop and 2-hop information
#
# 3. **2-Hop Implementation by Filter Type**:
#    - User filter: User → Users → Users → Items
#    - Item filter: User → Items → Items → Items  
#    - Bipartite filter: User ↔ Items ↔ Users ↔ Items
#    - Direct: User → Items → Users → Items
#
# 4. **Hop Decay Mechanism**:
#    - 2-hop components initialized with decay factor (default 0.5)
#    - Learnable weights balance 1-hop vs 2-hop contributions
#    - Prevents over-emphasis on distant neighbors
#
# 5. **Enhanced Configuration**:
#    - n_hops: 1 or 2 (configurable)
#    - hop_decay: Weight initialization for 2-hop components
#    - Automatic cache key generation includes hop information
#
# 🎯 USAGE EXAMPLES:
# 
# # 2-hop user-specific filtering
# python main.py --model_type user_specific --dataset ml-100k \
#     --filter ui --n_hops 2 --hop_decay 0.6 \
#     --u_n_eigen 30 --i_n_eigen 50 --shared_base
#
# # 2-hop user + bipartite filtering
# python main.py --model_type user_specific --dataset ml-100k \
#     --filter ub --n_hops 2 --hop_decay 0.4 \
#     --u_n_eigen 30 --b_n_eigen 40 --personalization_dim 16
#
# 📊 EXPECTED IMPROVEMENTS:
# - Better capture of extended user neighborhoods
# - Enhanced cold-start performance through 2-hop connections
# - Richer personalized representations
# - Improved handling of sparse interaction data
#
# 🔧 PARAMETERS TO TUNE:
# - hop_decay: 0.3-0.7 (lower = less emphasis on 2-hop)
# - personalization_dim: 8-32 (higher for more complex datasets)
# - Learning rates: May need adjustment for multi-hop training
#
# ============================================================================