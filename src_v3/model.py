'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Updated with separate u_n_eigen and i_n_eigen support + Comprehensive Caching

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
import world

# class UniversalSpectralFilter(nn.Module):
#     def __init__(self, filter_order=3):
#         super().__init__()
#         self.filter_order = filter_order

#         # Initialize coefficients based on a smooth low-pass filter
#         smooth_lowpass = [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015, -0.00003]
#         coeffs_data = torch.zeros(filter_order + 1)
#         for i, val in enumerate(smooth_lowpass[:filter_order + 1]):
#             coeffs_data[i] = val
        
#         self.coeffs = nn.Parameter(coeffs_data)
    
#     def forward(self, eigenvalues):
#         """Apply learnable spectral filter using Chebyshev polynomials"""
#         # Normalize eigenvalues to [-1, 1]
#         max_eigenval = torch.max(eigenvalues) + 1e-8
#         x = 2 * (eigenvalues / max_eigenval) - 1
        
#         # Compute Chebyshev polynomial response
#         result = self.coeffs[0] * torch.ones_like(x)
        
#         if len(self.coeffs) > 1:
#             T_prev, T_curr = torch.ones_like(x), x
#             result += self.coeffs[1] * T_curr
            
#             for i in range(2, len(self.coeffs)):
#                 T_next = 2 * x * T_curr - T_prev
#                 result += self.coeffs[i] * T_next
#                 T_prev, T_curr = T_curr, T_next
        
#         return torch.exp(-torch.abs(result)) + 1e-6

import torch
import torch.nn as nn

# Chebyshev filter
# filter_cheb = UniversalSpectralFilter(filter_order=4, basis_type="chebyshev")

# Legendre filter
# filter_leg = UniversalSpectralFilter(filter_order=4, basis_type="legendre")

# Jacobi filter (e.g., alpha=0.5, beta=0.5)
# filter_jacobi = UniversalSpectralFilter(filter_order=4, basis_type="jacobi", alpha=0.5, beta=0.5)

class UniversalSpectralFilter(nn.Module):
    def __init__(self, filter_order=3, basis_type="jacobi", alpha=0.5, beta=0.5):
        """
        basis_type: 'chebyshev', 'legendre', or 'jacobi'
        alpha, beta: only used for Jacobi polynomials
        """
        super().__init__()
        self.filter_order = filter_order
        self.basis_type = basis_type.lower()
        self.alpha = alpha
        self.beta = beta

        # Initialize coefficients from a smooth low-pass filter prior
        smooth_lowpass = [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015, -0.00003]
        coeffs_data = torch.zeros(filter_order + 1)
        for i, val in enumerate(smooth_lowpass[:filter_order + 1]):
            coeffs_data[i] = val
        
        self.coeffs = nn.Parameter(coeffs_data)

    def forward(self, eigenvalues):
        """
        Apply learnable spectral filter on eigenvalues using selected polynomial basis.
        Returns a smoothed filter response.
        """
        # Normalize eigenvalues to [-1, 1]
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1

        basis = self._compute_polynomials(x)
        result = sum(self.coeffs[i] * basis[i] for i in range(self.filter_order + 1))

        return torch.exp(-torch.abs(result)) + 1e-6

    def _compute_polynomials(self, x):
        """
        Computes the polynomial basis [P_0(x), ..., P_k(x)] depending on the selected type.
        """
        basis = []
        if self.basis_type == "chebyshev":
            T_prev, T_curr = torch.ones_like(x), x
            basis.append(T_prev)
            if self.filter_order >= 1:
                basis.append(T_curr)
            for _ in range(2, self.filter_order + 1):
                T_next = 2 * x * T_curr - T_prev
                basis.append(T_next)
                T_prev, T_curr = T_curr, T_next

        elif self.basis_type == "legendre":
            P_prev, P_curr = torch.ones_like(x), x
            basis.append(P_prev)
            if self.filter_order >= 1:
                basis.append(P_curr)
            for n in range(2, self.filter_order + 1):
                P_next = ((2 * n - 1) * x * P_curr - (n - 1) * P_prev) / n
                basis.append(P_next)
                P_prev, P_curr = P_curr, P_next

        elif self.basis_type == "jacobi":
            P_prev = torch.ones_like(x)
            basis.append(P_prev)
            if self.filter_order >= 1:
                P_curr = 0.5 * (2 * (self.alpha + 1) + (self.alpha + self.beta + 2) * (x - 1))
                basis.append(P_curr)

            for n in range(2, self.filter_order + 1):
                a1 = 2 * n * (n + self.alpha + self.beta) * (2 * n + self.alpha + self.beta - 2)
                a2 = (2 * n + self.alpha + self.beta - 1) * (self.alpha ** 2 - self.beta ** 2)
                a3 = (2 * n + self.alpha + self.beta - 2) * (2 * n + self.alpha + self.beta - 1) * (2 * n + self.alpha + self.beta)
                a4 = 2 * (n + self.alpha - 1) * (n + self.beta - 1) * (2 * n + self.alpha + self.beta)

                P_next = ((a2 + a3 * x) * P_curr - a4 * P_prev) / a1
                basis.append(P_next)
                P_prev, P_curr = P_curr, P_next

        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")

        return basis


class UniversalSpectralCF(nn.Module):
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.filter_order = self.config.get('filter_order', 3)
        self.filter = self.config.get('filter', 'ui')
        
        # Enhanced eigenvalue configuration with separate user/item counts
        self.u_n_eigen, self.i_n_eigen = self._get_eigenvalue_counts()
        
        # Convert and register adjacency matrix
        adj_dense = adj_mat.toarray() if sp.issparse(adj_mat) else adj_mat
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Compute and register normalized adjacency
        row_sums = self.adj_tensor.sum(dim=1, keepdim=True) + 1e-8
        col_sums = self.adj_tensor.sum(dim=0, keepdim=True) + 1e-8
        norm_adj = self.adj_tensor / torch.sqrt(row_sums) / torch.sqrt(col_sums)
        self.register_buffer('norm_adj', norm_adj)
        
        print(f"🔧 Universal Spectral CF with Caching:")
        print(f"   └─ Dataset: {self.config.get('dataset', 'unknown')}")
        print(f"   └─ Users: {self.n_users:,}, Items: {self.n_items:,}")
        print(f"   └─ User eigenvalues (u_n_eigen): {self.u_n_eigen}")
        print(f"   └─ Item eigenvalues (i_n_eigen): {self.i_n_eigen}")
        print(f"   └─ Filter Type: {self.filter}")
        print(f"   └─ Device: {self.device}")
        
        # Initialize filters and weights
        self._setup_filters()
        self._setup_combination_weights()
    
    def _get_eigenvalue_counts(self):
        """Get separate eigenvalue counts for users and items"""
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
        
        # Default values if not specified
        u_n_eigen = 50
        i_n_eigen = 50
        print(f"   🤖 Using default eigenvalue counts: u_n_eigen={u_n_eigen}, i_n_eigen={i_n_eigen}")
        return u_n_eigen, i_n_eigen
    
    def _get_cache_path(self, cache_type, filter_type=None):
        """Generate cache file path with relevant parameters"""
        cache_dir = "../cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = self.config.get('dataset', 'unknown')
        
        if filter_type:
            if cache_type.startswith('similarity'):
                # Similarity matrices only depend on dataset and data, not on eigen counts
                filename = f"{dataset}_universal_{filter_type}_sim.pkl"
            else:  # eigen
                # Eigendecompositions depend on eigen counts and other parameters
                filter_order = self.config.get('filter_order', 3)
                filter_mode = self.config.get('filter', 'ui')
                u_eigen = self.u_n_eigen
                i_eigen = self.i_n_eigen
                k_value = u_eigen if filter_type == 'user' else i_eigen
                filename = f"{dataset}_universal_u{u_eigen}_i{i_eigen}_fo{filter_order}_{filter_mode}_{filter_type}_eigen_k{k_value}.pkl"
        else:
            # Other cache types
            filter_order = self.config.get('filter_order', 3)
            filter_mode = self.config.get('filter', 'ui')
            u_eigen = self.u_n_eigen
            i_eigen = self.i_n_eigen
            filename = f"{dataset}_universal_u{u_eigen}_i{i_eigen}_fo{filter_order}_{filter_mode}_{cache_type}.pkl"
            
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
    
    def _memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _setup_filters(self):
        """Setup spectral filters with eigendecompositions"""
        print(f"Computing eigendecompositions for filter type: {self.filter}")
        start = time.time()
        
        # Initialize filters
        self.user_filter = None
        self.item_filter = None
        
        if self.filter in ['u', 'ui']:
            print("Processing user-user similarity matrix...")
            self.user_filter = self._create_filter('user')
            self._memory_cleanup()
        
        if self.filter in ['i', 'ui']:
            print("Processing item-item similarity matrix...")
            self.item_filter = self._create_filter('item')
            self._memory_cleanup()
        
        print(f'Filter setup completed in {time.time() - start:.2f}s')
    
    def _compute_similarity_matrix(self, interaction_matrix, cache_type=None):
        """Compute similarity matrix with caching"""
        
        # Try to load from cache first
        if cache_type:
            cache_path = self._get_cache_path('similarity', cache_type)
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data.to(self.device)
        
        print(f"    Computing similarity matrix...")
        
        # Compute similarity matrix (using simple dot product similarity)
        similarity = interaction_matrix @ interaction_matrix.t()
        
        # Normalize to get cosine similarity
        norms = torch.norm(interaction_matrix, dim=1, keepdim=True) + 1e-8
        similarity = similarity / (norms @ norms.t())
        
        # Ensure symmetry
        similarity = (similarity + similarity.t()) / 2
        
        # Set diagonal to 1 (self-similarity)
        similarity.fill_diagonal_(1.0)
        result = torch.clamp(similarity, min=0.0, max=1.0)
        
        # Save to cache
        if cache_type:
            self._save_to_cache(result.cpu(), cache_path)
        
        return result
    
    def _create_filter(self, filter_type):
        """Create filter with eigendecomposition using specified eigenvalue count"""
        
        # Use appropriate eigenvalue count
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
                    similarity_matrix = self._compute_similarity_matrix(self.norm_adj, cache_type='user')
                else:  # item
                    similarity_matrix = self._compute_similarity_matrix(self.norm_adj.t(), cache_type='item')
            
            print(f"  Computing eigendecomposition...")
            sim_np = similarity_matrix.cpu().numpy()
            
            del similarity_matrix
            self._memory_cleanup()
            
            k = min(n_eigen_to_use, n_components - 2)
            
            try:
                print(f"  Computing {k} largest eigenvalues for {filter_type}...")
                eigenvals, eigenvecs = eigsh(sp.csr_matrix(sim_np), k=k, which='LM')
                
                eigenvals_tensor = torch.tensor(np.real(eigenvals), dtype=torch.float32)
                eigenvecs_tensor = torch.tensor(np.real(eigenvecs), dtype=torch.float32)
                
                # Save to cache
                self._save_to_cache((eigenvals_tensor, eigenvecs_tensor), eigen_cache_path)
                
                # Register buffers
                self.register_buffer(f'{filter_type}_eigenvals', eigenvals_tensor.to(self.device))
                self.register_buffer(f'{filter_type}_eigenvecs', eigenvecs_tensor.to(self.device))
                
                print(f"  {filter_type.capitalize()} eigendecomposition: {k} components")
                print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
                
            except Exception as e:
                print(f"  {filter_type.capitalize()} eigendecomposition failed: {e}")
                print(f"  Using fallback identity matrices...")
                
                eigenvals = np.ones(min(n_eigen_to_use, n_components))
                eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
                eigenvecs_tensor = torch.eye(n_components, min(n_eigen_to_use, n_components))
                
                # Save fallback to cache
                self._save_to_cache((eigenvals_tensor, eigenvecs_tensor), eigen_cache_path)
                
                self.register_buffer(f'{filter_type}_eigenvals', eigenvals_tensor.to(self.device))
                self.register_buffer(f'{filter_type}_eigenvecs', eigenvecs_tensor.to(self.device))
            
            del sim_np
            if 'eigenvals' in locals():
                del eigenvals, eigenvecs
            self._memory_cleanup()
        
        return UniversalSpectralFilter(self.filter_order)
    
    def _setup_combination_weights(self):
        """Setup learnable combination weights"""
        init_weights = {
            'u': [0.5, 0.5],
            'i': [0.5, 0.5], 
            'ui': [0.5, 0.3, 0.2]
        }
        self.combination_weights = nn.Parameter(torch.tensor(init_weights[self.filter]))
    
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
        """Clean forward pass - ONLY returns predictions"""
        # Ensure users tensor is on correct device
        if users.device != self.adj_tensor.device:
            users = users.to(self.adj_tensor.device)
        
        user_profiles = self.adj_tensor[users]
        user_filter_matrix, item_filter_matrix = self._get_filter_matrices()
        
        # Compute filtered scores based on filter type
        scores = [user_profiles]  # Direct scores always included
        
        if self.filter in ['i', 'ui'] and item_filter_matrix is not None:
            scores.append(user_profiles @ item_filter_matrix)
        
        if self.filter in ['u', 'ui'] and user_filter_matrix is not None:
            user_filtered = user_filter_matrix[users] @ self.adj_tensor
            scores.append(user_filtered)
        
        # Combine scores with learned weights
        weights = torch.softmax(self.combination_weights, dim=0)
        predicted = sum(w * score for w, score in zip(weights, scores))
        
        # Memory cleanup for large datasets
        if self.training and (self.n_users > 5000 or self.n_items > 5000):
            del user_filter_matrix, item_filter_matrix
            self._memory_cleanup()
        
        return predicted  # ALWAYS return predictions only!
    
    def getUsersRating(self, batch_users):
        """Evaluation interface"""
        self.eval()
        with torch.no_grad():
            if isinstance(batch_users, np.ndarray):
                batch_users = torch.LongTensor(batch_users)
            
            # Ensure batch_users is on the correct device
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
        """Clear cache files for this dataset and configuration"""
        cache_dir = "../cache"
        if not os.path.exists(cache_dir):
            return
        
        dataset = self.config.get('dataset', 'unknown')
        
        # Look for files matching this dataset
        pattern_parts = [dataset, 'universal']
        
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
    
    def clear_similarity_cache(self):
        """Clear only similarity matrix cache files (when changing similarity computation)"""
        cache_dir = "../cache"
        if not os.path.exists(cache_dir):
            return
        
        dataset = self.config.get('dataset', 'unknown')
        
        # Look for similarity files only
        pattern_parts = [dataset, 'universal', 'sim.pkl']
        
        removed_count = 0
        for filename in os.listdir(cache_dir):
            if all(part in filename for part in pattern_parts):
                file_path = os.path.join(cache_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"🗑️ Removed similarity cache: {filename}")
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to remove {filename}: {e}")
        
        if removed_count == 0:
            print("No similarity cache files found")
        else:
            print(f"Removed {removed_count} similarity cache files")
    
    def clear_eigen_cache(self):
        """Clear only eigendecomposition cache files (when changing eigen counts)"""
        cache_dir = "../cache"
        if not os.path.exists(cache_dir):
            return
        
        dataset = self.config.get('dataset', 'unknown')
        
        # Look for eigen files only
        pattern_parts = [dataset, 'universal', 'eigen']
        
        removed_count = 0
        for filename in os.listdir(cache_dir):
            if all(part in filename for part in pattern_parts):
                file_path = os.path.join(cache_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"🗑️ Removed eigen cache: {filename}")
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to remove {filename}: {e}")
        
        if removed_count == 0:
            print("No eigen cache files found")
        else:
            print(f"Removed {removed_count} eigen cache files")

    def debug_filter_learning(self):
        """Debug what the filters are learning and identify filter patterns"""
        print("\n=== FILTER LEARNING DEBUG (WITH CACHING) ===")
        
        # Known filter patterns for comparison
        filter_patterns = {
            'butterworth': [1.0, -0.6, 0.2, -0.05, 0.01, -0.002, 0.0003, -0.00005],
            'chebyshev': [1.0, -0.4, 0.1, -0.01, 0.001, -0.0001, 0.00001, -0.000001],
            'smooth': [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015, -0.00003],
            'bessel': [1.0, -0.3, 0.06, -0.008, 0.0008, -0.00006, 0.000004, -0.0000002],
            'gaussian': [1.0, -0.7, 0.15, -0.03, 0.005, -0.0007, 0.00008, -0.000008],
            'conservative': [1.0, -0.2, 0.03, -0.002, 0.0001, -0.000005, 0.0000002, -0.00000001],
            'aggressive': [1.0, -0.8, 0.3, -0.08, 0.015, -0.002, 0.0002, -0.00002]
        }
        
        def analyze_filter_pattern(coeffs_tensor, filter_name):
            """Analyze learned coefficients and find closest pattern"""
            coeffs = coeffs_tensor.cpu().numpy()
            print(f"\n{filter_name} Filter Analysis:")
            print(f"  Learned coefficients: {coeffs}")
            
            # Find closest pattern
            best_match = None
            best_similarity = -1
            
            for pattern_name, pattern_coeffs in filter_patterns.items():
                # Compare with same number of coefficients
                pattern_truncated = pattern_coeffs[:len(coeffs)]
                
                # Calculate correlation coefficient
                if len(coeffs) > 1 and len(pattern_truncated) > 1:
                    correlation = np.corrcoef(coeffs, pattern_truncated)[0, 1]
                    if not np.isnan(correlation) and correlation > best_similarity:
                        best_similarity = correlation
                        best_match = pattern_name
            
            # Determine filter characteristics
            filter_type = classify_filter_behavior(coeffs)
            
            print(f"  📊 Filter Characteristics:")
            print(f"     └─ Type: {filter_type}")
            print(f"     └─ Closest pattern: {best_match} (similarity: {best_similarity:.3f})")
            
            # Pattern interpretation
            if best_similarity > 0.9:
                print(f"     └─ 🎯 Strong match to {best_match} filter!")
            elif best_similarity > 0.7:
                print(f"     └─ ✅ Good match to {best_match}-like behavior")
            elif best_similarity > 0.5:
                print(f"     └─ 🔄 Moderate similarity to {best_match}")
            else:
                print(f"     └─ 🆕 Learned unique pattern (not matching standard filters)")
            
            return best_match, best_similarity, filter_type
        
        def classify_filter_behavior(coeffs):
            """Classify the learned filter behavior"""
            if len(coeffs) < 2:
                return "constant"
            
            # Analyze coefficient pattern
            c0, c1 = coeffs[0], coeffs[1]
            
            # Check for different behaviors
            if abs(c0) > 0.8 and c1 < -0.3:
                if len(coeffs) > 2 and coeffs[2] > 0:
                    return "low-pass (strong)"
                else:
                    return "low-pass (moderate)"
            elif abs(c0) < 0.3 and c1 > 0.3:
                return "high-pass"
            elif c0 > 0.5 and abs(c1) < 0.3:
                return "conservative low-pass"
            elif len(coeffs) > 2 and abs(coeffs[2]) > 0.1:
                return "band-pass/complex"
            else:
                return "custom/mixed"
        
        print(f"Cache Status:")
        print(f"  └─ Cache directory: ../cache")
        print(f"  └─ User eigenvalues: {self.u_n_eigen} (cached separately)")
        print(f"  └─ Item eigenvalues: {self.i_n_eigen} (cached separately)")
        
        with torch.no_grad():
            # Analyze user filter
            if self.filter in ['u', 'ui'] and self.user_filter is not None:
                user_match, user_sim, user_type = analyze_filter_pattern(
                    self.user_filter.coeffs, f"User ({self.u_n_eigen} eigenvalues)"
                )
                user_response = self.user_filter(self.user_eigenvals)
                print(f"  Filter response range: [{user_response.min():.4f}, {user_response.max():.4f}]")
            
            # Analyze item filter
            if self.filter in ['i', 'ui'] and self.item_filter is not None:
                item_match, item_sim, item_type = analyze_filter_pattern(
                    self.item_filter.coeffs, f"Item ({self.i_n_eigen} eigenvalues)"
                )
                item_response = self.item_filter(self.item_eigenvals)
                print(f"  Filter response range: [{item_response.min():.4f}, {item_response.max():.4f}]")
            
            # Combination weights analysis
            weights = torch.softmax(self.combination_weights, dim=0)
            print(f"\n🔗 Combination Weights Analysis:")
            print(f"  Raw weights: {weights.cpu().numpy()}")
            
            if self.filter == 'ui':
                direct, item, user = weights.cpu().numpy()
                print(f"  📈 Component Importance:")
                print(f"     └─ Direct CF: {direct:.3f} ({'🔥 Dominant' if direct > 0.5 else '🔸 Moderate' if direct > 0.3 else '🔹 Minor'})")
                print(f"     └─ Item filtering: {item:.3f} ({'🔥 Dominant' if item > 0.5 else '🔸 Moderate' if item > 0.3 else '🔹 Minor'})")
                print(f"     └─ User filtering: {user:.3f} ({'🔥 Dominant' if user > 0.5 else '🔸 Moderate' if user > 0.3 else '🔹 Minor'})")
            elif self.filter == 'u':
                direct, user = weights.cpu().numpy()
                print(f"     └─ Direct CF: {direct:.3f}")
                print(f"     └─ User filtering: {user:.3f}")
            elif self.filter == 'i':
                direct, item = weights.cpu().numpy()
                print(f"     └─ Direct CF: {direct:.3f}")
                print(f"     └─ Item filtering: {item:.3f}")
            
            # Overall model interpretation
            print(f"\n🎯 Overall Model Interpretation:")
            if self.filter == 'ui':
                if 'user_match' in locals() and 'item_match' in locals():
                    print(f"  └─ User-side learned: {user_type} ({user_match}-like)")
                    print(f"  └─ Item-side learned: {item_type} ({item_match}-like)")
                    
                    # Suggest what this means
                    if user_type.startswith('low-pass') and item_type.startswith('low-pass'):
                        print(f"  🔍 Model focuses on global patterns (popular items, broad preferences)")
                    elif 'high-pass' in user_type or 'high-pass' in item_type:
                        print(f"  🔍 Model emphasizes niche patterns (specific preferences)")
                    else:
                        print(f"  🔍 Model learned balanced filtering strategy")
            
        print("=== END DEBUG ===\n")