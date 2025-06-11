'''
Created on June 7, 2025
DySimGCF-Style Universal Spectral CF with Similarity-Aware Laplacian
Enhanced with separate u_n_eigen and i_n_eigen for optimal eigendecomposition

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
    """DySimGCF-Style Universal Spectral CF with Separate User/Item Eigenvalue Counts"""
    
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.filter_order = self.config.get('filter_order', 6)
        self.filter = self.config.get('filter', 'ui')
        
        # Convert and get dataset characteristics first
        adj_dense = adj_mat.toarray() if sp.issparse(adj_mat) else adj_mat
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32).to(self.device))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Calculate dataset characteristics
        total_interactions = torch.sum(self.adj_tensor).item()
        sparsity = total_interactions / (self.n_users * self.n_items)
        
        # Separate adaptive eigenvalue calculation for users and items
        self.u_n_eigen, self.i_n_eigen = self._calculate_adaptive_eigenvalues(total_interactions, sparsity)
        
        # Similarity parameters
        self.similarity_type = self.config.get('similarity_type', 'cosine')
        self.similarity_threshold = self.config.get('similarity_threshold', 0.01)
        
        # Filter design selection
        self.filter_design = self.config.get('filter_design', 'enhanced_basis')
        self.init_filter = self.config.get('init_filter', 'smooth')
        
        print(f"🔧 DySimGCF-Style Spectral CF with Separate Eigenvalues:")
        print(f"   └─ Dataset: {self.config.get('dataset', 'unknown')}")
        print(f"   └─ Users: {self.n_users:,}, Items: {self.n_items:,}")
        print(f"   └─ Interactions: {int(total_interactions):,}")
        print(f"   └─ Sparsity: {sparsity:.4f}")
        print(f"   └─ User eigenvalues (u_n_eigen): {self.u_n_eigen}")
        print(f"   └─ Item eigenvalues (i_n_eigen): {self.i_n_eigen}")
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
        """Calculate separate adaptive eigenvalues for users and items"""
        
        # Manual override if specified
        manual_u_eigen = self.config.get('u_n_eigen', None)
        manual_i_eigen = self.config.get('i_n_eigen', None)
        manual_n_eigen = self.config.get('n_eigen', None)
        
        if manual_u_eigen is not None and manual_i_eigen is not None:
            if manual_u_eigen > 0 and manual_i_eigen > 0:
                print(f"   🎯 Using manual u_n_eigen: {manual_u_eigen}, i_n_eigen: {manual_i_eigen}")
                return manual_u_eigen, manual_i_eigen
        elif manual_n_eigen is not None and manual_n_eigen > 0:
            print(f"   🎯 Using manual n_eigen for both: {manual_n_eigen}")
            return manual_n_eigen, manual_n_eigen
        
        # Separate calculations for users and items
        u_n_eigen = self._calculate_matrix_eigenvalues(
            self.n_users, total_interactions, sparsity, "user"
        )
        i_n_eigen = self._calculate_matrix_eigenvalues(
            self.n_items, total_interactions, sparsity, "item"
        )
        
        return u_n_eigen, i_n_eigen
    
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
            # Users typically have fewer interactions per entity
            avg_interactions_per_user = total_interactions / self.n_users
            if avg_interactions_per_user < 10:
                user_multiplier = 1.3  # Need more eigenvalues for sparse user patterns
            elif avg_interactions_per_user > 100:
                user_multiplier = 0.8  # Can use fewer for dense user patterns
            else:
                user_multiplier = 1.0
                
        else:  # item
            # Items typically have different popularity distributions
            avg_interactions_per_item = total_interactions / self.n_items
            if avg_interactions_per_item < 5:
                item_multiplier = 1.4  # Long tail items need more eigenvalues
            elif avg_interactions_per_item > 50:
                item_multiplier = 0.7  # Popular items can use fewer
            else:
                item_multiplier = 1.0
        
        multiplier = user_multiplier if matrix_type == "user" else item_multiplier
        
        # Sparsity adjustment (same for both)
        if sparsity < 0.001:  # Very sparse
            sparsity_multiplier = 1.4
        elif sparsity < 0.01:  # Sparse
            sparsity_multiplier = 1.2
        elif sparsity > 0.05:  # Dense
            sparsity_multiplier = 0.8
        else:
            sparsity_multiplier = 1.0
        
        # Calculate final eigenvalue count
        adaptive_eigen = int(base_eigen * multiplier * sparsity_multiplier)
        
        # Ensure reasonable bounds
        min_eigen = 16
        max_eigen = min(384, matrix_size - 10)  # Matrix size constraint
        
        adaptive_eigen = max(min_eigen, min(adaptive_eigen, max_eigen))
        
        print(f"   📊 {matrix_type.capitalize()} eigenvalue calculation:")
        print(f"      Matrix size: {matrix_size}")
        print(f"      Base: {base_eigen}")
        print(f"      {matrix_type.capitalize()} mult: {multiplier:.2f}")
        print(f"      Sparsity mult: {sparsity_multiplier:.2f}")
        print(f"      Final: {adaptive_eigen}")
        
        return adaptive_eigen
    
    def _get_cache_path(self, cache_type, filter_type=None):
        """FIXED: Generate cache file path with ALL relevant parameters"""
        cache_dir = "../cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = self.config.get('dataset', 'unknown')
        sim_type = self.similarity_type
        threshold = str(self.similarity_threshold).replace('.', 'p')
        
        # CRITICAL FIX: Include ALL parameters that affect eigendecomposition
        u_eigen = self.u_n_eigen
        i_eigen = self.i_n_eigen
        filter_design = self.config.get('filter_design', 'enhanced_basis')
        init_filter = self.config.get('init_filter', 'smooth')
        filter_order = self.config.get('filter_order', 6)
        filter_mode = self.config.get('filter', 'ui')
        
        # Create comprehensive filename
        base_name = f"{dataset}_{sim_type}_th{threshold}_u{u_eigen}_i{i_eigen}_{filter_design}_{init_filter}_fo{filter_order}_{filter_mode}"
        
        if filter_type:
            if cache_type.startswith('similarity'):
                filename = f"{base_name}_{filter_type}_sim.pkl"
            else:  # eigen
                k_value = u_eigen if filter_type == 'user' else i_eigen
                filename = f"{base_name}_{filter_type}_eigen_k{k_value}.pkl"
        else:
            filename = f"{base_name}_{cache_type}.pkl"
            
        return os.path.join(cache_dir, filename)
    
    @staticmethod
    def get_dataset_recommendations(dataset_name, n_users, n_items, sparsity):
        """Get dataset-specific recommendations with separate user/item eigenvalues"""
        
        recommendations = {
            'u_n_eigen': 'auto',
            'i_n_eigen': 'auto',
            'filter_design': 'enhanced_basis',
            'similarity_type': 'cosine',
            'similarity_threshold': 0.01,
            'reasoning': []
        }
        
        # Dataset-specific optimizations
        if 'ml-100k' in dataset_name.lower():
            recommendations.update({
                'u_n_eigen': 48,  # 943 users
                'i_n_eigen': 64,  # 1682 items
                'filter_design': 'enhanced_basis',
                'similarity_threshold': 0.02,
                'reasoning': ['Small dataset - moderate eigenvalue counts', 'More item eigenvalues for item diversity']
            })
        elif 'ml-1m' in dataset_name.lower():
            recommendations.update({
                'u_n_eigen': 96,   # ~6000 users
                'i_n_eigen': 128,  # ~4000 items
                'filter_design': 'enhanced_basis',
                'similarity_threshold': 0.01,
                'reasoning': ['Medium dataset - balanced eigenvalue allocation']
            })
        elif 'lastfm' in dataset_name.lower():
            recommendations.update({
                'u_n_eigen': 64,   # 1892 users
                'i_n_eigen': 96,   # 4489 items
                'filter_design': 'enhanced_basis',
                'similarity_threshold': 0.008,
                'reasoning': ['Music domain - more item eigenvalues for genre diversity']
            })
        elif 'gowalla' in dataset_name.lower():
            recommendations.update({
                'u_n_eigen': 128,  # Many users
                'i_n_eigen': 256,  # Many locations
                'filter_design': 'multiscale',
                'similarity_threshold': 0.005,
                'reasoning': ['Location data - high item eigenvalues for geographic patterns']
            })
        elif 'yelp' in dataset_name.lower():
            recommendations.update({
                'u_n_eigen': 192,  # Many users
                'i_n_eigen': 384,  # Many businesses
                'filter_design': 'multiscale',
                'similarity_type': 'jaccard',
                'similarity_threshold': 0.003,
                'reasoning': ['Large sparse dataset - high eigenvalue counts', 'More item eigenvalues for business diversity']
            })
        elif 'amazon' in dataset_name.lower():
            recommendations.update({
                'u_n_eigen': 256,  # Very many users
                'i_n_eigen': 512,  # Huge item catalog
                'filter_design': 'ensemble',
                'similarity_type': 'jaccard',
                'similarity_threshold': 0.001,
                'reasoning': ['Extremely large dataset', 'Very high item eigenvalues for product diversity']
            })
        
        # User/Item ratio adjustments
        user_item_ratio = n_users / n_items
        if user_item_ratio > 2.0:  # More users than items
            recommendations['reasoning'].append('More users than items - balanced eigenvalue allocation')
        elif user_item_ratio < 0.5:  # More items than users
            recommendations['reasoning'].append('More items than users - increased item eigenvalues')
        
        return recommendations
    
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
        """Compute symmetric similarity matrix with simple caching"""
        
        # Try to load from cache first
        if cache_type:
            cache_path = self._get_cache_path('similarity', cache_type)
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data.to(self.device)
        
        print(f"    Computing {self.similarity_type} similarity...")
        
        if self.similarity_type == 'cosine':
            # Cosine similarity: A @ A.T / (||A|| * ||A||)
            norms = torch.norm(interaction_matrix, dim=1, keepdim=True) + 1e-8
            normalized_matrix = interaction_matrix / norms
            similarity = normalized_matrix @ normalized_matrix.t()
        
        elif self.similarity_type == 'jaccard':
            # Jaccard similarity: |A ∩ B| / |A ∪ B|
            intersection = interaction_matrix @ interaction_matrix.t()
            sum_matrix = interaction_matrix.sum(dim=1, keepdim=True)
            union = sum_matrix + sum_matrix.t() - intersection
            similarity = intersection / (union + 1e-8)
        
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
        
        # Ensure symmetry and apply threshold
        similarity = (similarity + similarity.t()) / 2
        
        # Debug: Check similarity statistics before thresholding
        orig_nonzero = (similarity > 0).float().mean().item()
        orig_mean = similarity.mean().item()
        orig_max = similarity.max().item()
        
        # Auto-adaptive threshold if threshold is negative
        if self.similarity_threshold < 0:
            # Memory-efficient percentile calculation for large matrices
            if similarity.numel() > 10_000_000:  # 10M elements threshold
                print(f"    Large matrix ({similarity.shape}), using memory-efficient percentile...")
                # Sample-based percentile estimation
                sample_size = min(1_000_000, similarity.numel() // 10)
                flat_sim = similarity.flatten()
                indices = torch.randperm(flat_sim.numel())[:sample_size]
                sampled_similarities = flat_sim[indices]
                adaptive_threshold = torch.quantile(sampled_similarities, 0.7).item()
                print(f"    Sample-based adaptive threshold: {adaptive_threshold:.6f} (sampled {sample_size:,} elements)")
            else:
                # Standard percentile calculation for smaller matrices
                adaptive_threshold = torch.quantile(similarity.flatten(), 0.7).item()
                print(f"    Auto-adaptive threshold: {adaptive_threshold:.6f} (keeping top 30% similarities)")
        else:
            adaptive_threshold = self.similarity_threshold
        
        similarity_thresholded = torch.where(similarity >= adaptive_threshold, 
                                           similarity, torch.zeros_like(similarity))
        
        # Debug: Check similarity statistics after thresholding  
        thresh_nonzero = (similarity_thresholded > 0).float().mean().item()
        thresh_mean = similarity_thresholded.mean().item()
        
        print(f"    Similarity stats: orig_nonzero={orig_nonzero:.3f}, orig_mean={orig_mean:.4f}, orig_max={orig_max:.4f}")
        print(f"    After threshold {adaptive_threshold:.6f}: nonzero={thresh_nonzero:.3f}, mean={thresh_mean:.4f}")
        
        # Set diagonal to 1 (self-similarity)
        similarity_thresholded.fill_diagonal_(1.0)
        result = torch.clamp(similarity_thresholded, min=0.0, max=1.0)
        
        # Save to cache
        if cache_type:
            self._save_to_cache(result.cpu(), cache_path)
        
        return result
    
    def _compute_similarity_laplacian(self, similarity_matrix):
        """Compute normalized similarity Laplacian: L = I - D^(-1/2) * A * D^(-1/2)"""
        # Degree matrix: sum of similarity scores (not just connectivity count)
        degree = similarity_matrix.sum(dim=1) + 1e-8
        
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        # Create diagonal matrix and normalize
        normalized_laplacian = similarity_matrix * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
        
        # L = I - normalized_adjacency (standard normalized Laplacian)
        identity = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
        laplacian = identity - normalized_laplacian
        
        return laplacian
    
    def _setup_filters(self):
        """Setup spectral filters for user and item similarity Laplacians"""
        print(f"Computing similarity Laplacians for filter type: {self.filter}")
        start = time.time()
        
        self.user_filter = None
        self.item_filter = None
        
        if self.filter in ['u', 'ui']:
            print("Processing user-user similarity Laplacian...")
            self.user_filter = self._create_similarity_filter('user')
            self._memory_cleanup()
        
        if self.filter in ['i', 'ui']:
            print("Processing item-item similarity Laplacian...")
            self.item_filter = self._create_similarity_filter('item')
            self._memory_cleanup()
        
        print(f'DySimGCF-style filter setup completed in {time.time() - start:.2f}s')
    
    def _create_similarity_filter(self, filter_type):
        """Create spectral filter with appropriate eigenvalue count"""
        
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
        # NEW BAND-STOP AND ADVANCED FILTERS
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
        """Setup learnable combination weights"""
        init_weights = {
            'u': [0.5, 0.5],
            'i': [0.5, 0.5], 
            'ui': [0.5, 0.3, 0.2]
        }
        self.combination_weights = nn.Parameter(torch.tensor(init_weights[self.filter]).to(self.device))
    
    def _get_filter_matrices(self):
        """Compute spectral filter matrices"""
        user_matrix = item_matrix = None
        
        if self.user_filter is not None:
            response = self.user_filter(self.user_eigenvals)
            user_matrix = self.user_eigenvecs @ torch.diag(response) @ self.user_eigenvecs.t()
        
        if self.item_filter is not None:
            response = self.item_filter(self.item_eigenvals)
            item_matrix = self.item_eigenvecs @ torch.diag(response) @ self.item_eigenvecs.t()
        
        return user_matrix, item_matrix
    
    def forward(self, users):
        """Forward pass: DySimGCF-style separate user and item filtering"""
        # FIXED: Ensure users tensor is on the same device as adj_tensor
        if users.device != self.adj_tensor.device:
            users = users.to(self.adj_tensor.device)
        
        user_profiles = self.adj_tensor[users]
        user_filter_matrix, item_filter_matrix = self._get_filter_matrices()
        
        scores = [user_profiles]
        
        # Item-based filtering: user profiles through item similarity filter
        if self.filter in ['i', 'ui'] and item_filter_matrix is not None:
            scores.append(user_profiles @ item_filter_matrix)
        
        # User-based filtering: user similarity filter through interactions
        if self.filter in ['u', 'ui'] and user_filter_matrix is not None:
            user_filtered = user_filter_matrix[users] @ self.adj_tensor
            scores.append(user_filtered)
        
        # Combine predictions using learnable weights
        weights = torch.softmax(self.combination_weights, dim=0)
        predicted = sum(w * score for w, score in zip(weights, scores))
        
        if self.training and (self.n_users > 5000 or self.n_items > 5000):
            del user_filter_matrix, item_filter_matrix
            self._memory_cleanup()
        
        return predicted
    
    def getUsersRating(self, batch_users):
        """Evaluation interface"""
        self.eval()
        with torch.no_grad():
            if isinstance(batch_users, np.ndarray):
                batch_users = torch.LongTensor(batch_users)
            
            # FIXED: Ensure batch_users is on the correct device
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
        return filter_params
    
    def get_other_parameters(self):
        """Get non-filter parameters"""
        filter_param_ids = {id(p) for p in self.get_filter_parameters()}
        return [p for p in self.parameters() if id(p) not in filter_param_ids]
    
    def debug_filter_learning(self):
        """Debug similarity-aware spectral filtering with separate eigenvalue info"""
        print(f"\n=== DYSIMGCF-STYLE SPECTRAL FILTER DEBUG ===")
        print(f"Similarity Type: {self.similarity_type}")
        print(f"Similarity Threshold: {self.similarity_threshold}")
        print(f"Filter Design: {self.filter_design}")
        print(f"Device: {self.device}")
        print(f"User Eigenvalues: {self.u_n_eigen}")
        print(f"Item Eigenvalues: {self.i_n_eigen}")
        
        with torch.no_grad():
            if self.filter in ['u', 'ui'] and self.user_filter is not None:
                print(f"\n👤 User Similarity Laplacian Filter ({self.u_n_eigen} eigenvalues):")
                self._debug_single_filter(self.user_filter)
                
                if hasattr(self, 'user_eigenvals'):
                    eigenvals = self.user_eigenvals.cpu().numpy()
                    print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
                    print(f"  Eigenvalue mean: {eigenvals.mean():.4f}")
            
            if self.filter in ['i', 'ui'] and self.item_filter is not None:
                print(f"\n🎬 Item Similarity Laplacian Filter ({self.i_n_eigen} eigenvalues):")
                self._debug_single_filter(self.item_filter)
                
                if hasattr(self, 'item_eigenvals'):
                    eigenvals = self.item_eigenvals.cpu().numpy()
                    print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
                    print(f"  Eigenvalue mean: {eigenvals.mean():.4f}")
            
            weights = torch.softmax(self.combination_weights, dim=0)
            print(f"\n🔗 Combination Weights: {weights.cpu().numpy()}")
        
        print("=== END DEBUG ===\n")
    
    def _debug_single_filter(self, filter_obj):
        """Debug individual filter"""
        if isinstance(filter_obj, fl.UniversalSpectralFilter):
            init_coeffs = filter_obj.init_coeffs.cpu().numpy()
            current_coeffs = filter_obj.coeffs.cpu().numpy()
            change = current_coeffs - init_coeffs
            
            print(f"  Initial: {init_coeffs}")
            print(f"  Current: {current_coeffs}")
            print(f"  Change:  {change}")
            print(f"  Max |Δ|: {np.abs(change).max():.6f}")
            
        elif isinstance(filter_obj, (fl.SpectralBasisFilter, fl.EnhancedSpectralBasisFilter)):
            if hasattr(filter_obj, 'get_mixing_analysis'):
                mixing = filter_obj.get_mixing_analysis()
                print(f"  Top mixing weights:")
                for name, weight in list(mixing.items())[:3]:
                    print(f"    {name:15}: {weight:.4f}")
        
        else:
            param_count = sum(p.numel() for p in filter_obj.parameters())
            print(f"  Filter: {type(filter_obj).__name__} ({param_count} params)")

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
        """Clear cache files for this dataset and similarity settings"""
        cache_dir = "../cache"
        if not os.path.exists(cache_dir):
            return
        
        dataset = self.config.get('dataset', 'unknown')
        sim_type = self.similarity_type
        threshold = str(self.similarity_threshold).replace('.', 'p')
        
        # Look for files matching this configuration
        pattern_parts = [dataset, sim_type, f"th{threshold}"]
        
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
            print("No matching cache files found")
        else:
            print(f"Removed {removed_count} cache files")