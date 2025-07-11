'''
Created on June 12, 2025
Fixed Universal Spectral CF with Complete Filter Collection
Properly integrated with comprehensive filters.py

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

# Import simplified model
from model_simplified import SimplifiedSpectralCF


class UserSpecificUniversalSpectralCF(nn.Module):
    """Universal Spectral CF with Complete Filter Collection"""
    
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        # Basic configuration
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.filter_order = self.config.get('filter_order', 6)
        self.filter = self.config.get('filter', 'ui')
        self.dataset = self.config.get('dataset', 'unknown')
        self.n_hops = self.config.get('n_hops', 2)  # Number of hops for bipartite filtering
        self.hop_weight = self.config.get('hop_weight', 0.7)  # Weight for multi-hop vs direct
        
        # Convert adjacency matrix
        adj_dense = adj_mat.toarray() if sp.issparse(adj_mat) else adj_mat
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32).to(self.device))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Eigenvalue counts (fixed auto-calculation)
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
        
        # Setup filters and eigendecompositions
        self.setup_spectral_filters()
        self.setup_combination_weights()
    
    def setup_spectral_filters(self):
        """Main setup - compute eigendecompositions and setup filters"""
        start = time.time()
        
        # Setup view-specific filters
        if self.filter in ['u', 'ui', 'uib', 'ub']:
            print("Computing user-user similarity...")
            self.user_eigenvals, self.user_eigenvecs = self.compute_user_eigen()
            self.user_filter = self.create_user_filter()
        
        if self.filter in ['i', 'ui', 'uib']:
            print("Computing item-item similarity...")
            self.item_eigenvals, self.item_eigenvecs = self.compute_item_eigen()
            self.item_filter = self.create_item_filter()
        
        if self.filter in ['b', 'uib', 'ub']:
            print("Computing bipartite structure...")
            self.bipartite_eigenvals, self.bipartite_eigenvecs = self.compute_bipartite_eigen()
            self.bipartite_filter = self.create_bipartite_filter()
        
        print(f'Training completed in {time.time() - start:.2f}s')
    
    def compute_user_eigen(self):
        """Compute user similarity eigendecomposition"""
        cache_path = f"../cache/{self.dataset}_user_eigen_{self.u_n_eigen}.pkl"
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                eigenvals, eigenvecs = pickle.load(f)
            return eigenvals.to(self.device), eigenvecs.to(self.device)
        
        # GF-CF style normalization
        # Row normalization (user normalization)
        rowsum = self.adj_tensor.sum(dim=1)
        d_inv = torch.pow(rowsum + 1e-8, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat = torch.diag(d_inv)
        norm_adj = d_mat @ self.adj_tensor
        
        # Column normalization  
        colsum = self.adj_tensor.sum(dim=0)
        d_inv = torch.pow(colsum + 1e-8, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat = torch.diag(d_inv)
        norm_adj = norm_adj @ d_mat
        
        # User-user similarity: UU^T
        user_sim = norm_adj @ norm_adj.t()
        laplacian = torch.eye(self.n_users, device=self.device) - user_sim
        
        # Eigendecomposition
        k = min(self.u_n_eigen, self.n_users - 2)
        eigenvals, eigenvecs = eigsh(sp.csr_matrix(laplacian.cpu().numpy()), k=k, which='SM')
        eigenvals = np.maximum(eigenvals, 0.0)
        
        eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
        eigenvecs_tensor = torch.tensor(eigenvecs, dtype=torch.float32)
        
        # Cache results
        os.makedirs("../cache", exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((eigenvals_tensor, eigenvecs_tensor), f)
        
        return eigenvals_tensor.to(self.device), eigenvecs_tensor.to(self.device)
    
    def compute_item_eigen(self):
        """Compute item similarity eigendecomposition"""
        cache_path = f"../cache/{self.dataset}_item_eigen_{self.i_n_eigen}.pkl"
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                eigenvals, eigenvecs = pickle.load(f)
            return eigenvals.to(self.device), eigenvecs.to(self.device)
        
        # GF-CF style normalization (same as user)
        # Row normalization (user normalization)
        rowsum = self.adj_tensor.sum(dim=1)
        d_inv = torch.pow(rowsum + 1e-8, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat = torch.diag(d_inv)
        norm_adj = d_mat @ self.adj_tensor
        
        # Column normalization
        colsum = self.adj_tensor.sum(dim=0)
        d_inv = torch.pow(colsum + 1e-8, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat = torch.diag(d_inv)
        norm_adj = norm_adj @ d_mat
        
        # Item-item similarity: U^TU
        item_sim = norm_adj.t() @ norm_adj
        laplacian = torch.eye(self.n_items, device=self.device) - item_sim
        
        # Eigendecomposition
        k = min(self.i_n_eigen, self.n_items - 2)
        eigenvals, eigenvecs = eigsh(sp.csr_matrix(laplacian.cpu().numpy()), k=k, which='SM')
        eigenvals = np.maximum(eigenvals, 0.0)
        
        eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
        eigenvecs_tensor = torch.tensor(eigenvecs, dtype=torch.float32)
        
        # Cache results
        os.makedirs("../cache", exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((eigenvals_tensor, eigenvecs_tensor), f)
        
        return eigenvals_tensor.to(self.device), eigenvecs_tensor.to(self.device)
    
    def compute_bipartite_eigen(self):
        """Compute bipartite eigendecomposition"""
        cache_path = f"../cache/{self.dataset}_bipartite_eigen_{self.b_n_eigen}.pkl"
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                eigenvals, eigenvecs = pickle.load(f)
            return eigenvals.to(self.device), eigenvecs.to(self.device)
        
        # GF-CF style bipartite normalization
        # Row normalization (user normalization)
        rowsum = self.adj_tensor.sum(dim=1)
        d_inv = torch.pow(rowsum + 1e-8, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat = torch.diag(d_inv)
        norm_adj = d_mat @ self.adj_tensor
        
        # Column normalization
        colsum = self.adj_tensor.sum(dim=0)
        d_inv = torch.pow(colsum + 1e-8, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat = torch.diag(d_inv)
        norm_adj = norm_adj @ d_mat
        
        # Create bipartite matrix [0, U; U^T, 0]
        n_total = self.n_users + self.n_items
        bipartite = torch.zeros(n_total, n_total, device=self.device)
        bipartite[:self.n_users, self.n_users:] = norm_adj
        bipartite[self.n_users:, :self.n_users] = norm_adj.t()
        
        laplacian = torch.eye(n_total, device=self.device) - bipartite
        
        # Eigendecomposition
        k = min(self.b_n_eigen, n_total - 2)
        eigenvals, eigenvecs = eigsh(sp.csr_matrix(laplacian.cpu().numpy()), k=k, which='SM')
        eigenvals = np.maximum(eigenvals, 0.0)
        
        eigenvals_tensor = torch.tensor(eigenvals, dtype=torch.float32)
        eigenvecs_tensor = torch.tensor(eigenvecs, dtype=torch.float32)
        
        # Cache results
        os.makedirs("../cache", exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((eigenvals_tensor, eigenvecs_tensor), f)
        
        return eigenvals_tensor.to(self.device), eigenvecs_tensor.to(self.device)
    
    def create_user_filter(self):
        """Create user-specific filter using complete filter collection"""
        user_design = self.config.get('user_filter_design', 'enhanced_basis')
        user_init = self.config.get('user_init_filter', 'smooth')
        user_dim = self.config.get('user_personalization_dim', 16)
        
        # Get additional parameters for advanced filters
        filter_kwargs = {
            'n_bands': self.config.get('n_bands', 4),
            'n_harmonics': self.config.get('n_harmonics', 3),
            'n_stop_bands': self.config.get('n_stop_bands', 2),
            'alpha': self.config.get('alpha', 0.0),
            'beta': self.config.get('beta', 0.0),
            'polynomial_type': self.config.get('polynomial_type', 'chebyshev'),
            'polynomial_params': {'alpha': self.config.get('alpha', 0.0), 'beta': self.config.get('beta', 0.0)}
        }
        
        return UserSpecificFilter(
            self.n_users, self.n_items, self.filter_order, 
            user_init, True, user_dim, user_design, **filter_kwargs
        )
    
    def create_item_filter(self):
        """Create item-specific filter using complete filter collection"""
        item_design = self.config.get('item_filter_design', 'chebyshev')
        item_init = self.config.get('item_init_filter', 'sharp')
        item_dim = self.config.get('item_personalization_dim', 12)
        
        # Get additional parameters for advanced filters
        filter_kwargs = {
            'n_bands': self.config.get('n_bands', 4),
            'n_harmonics': self.config.get('n_harmonics', 3),
            'n_stop_bands': self.config.get('n_stop_bands', 2),
            'alpha': self.config.get('alpha', 0.0),
            'beta': self.config.get('beta', 0.0),
            'polynomial_type': self.config.get('polynomial_type', 'chebyshev'),
            'polynomial_params': {'alpha': self.config.get('alpha', 0.0), 'beta': self.config.get('beta', 0.0)}
        }
        
        return ItemSpecificFilter(
            self.n_users, self.n_items, self.filter_order,
            item_init, True, item_dim, item_design, **filter_kwargs
        )
    
    def create_bipartite_filter(self):
        """Create bipartite-specific filter using complete filter collection"""
        bipartite_design = self.config.get('bipartite_filter_design', 'original')
        bipartite_init = self.config.get('bipartite_init_filter', 'smooth')
        bipartite_dim = self.config.get('bipartite_personalization_dim', 20)
        
        # Get additional parameters for advanced filters
        filter_kwargs = {
            'n_bands': self.config.get('n_bands', 4),
            'n_harmonics': self.config.get('n_harmonics', 3),
            'n_stop_bands': self.config.get('n_stop_bands', 2),
            'alpha': self.config.get('alpha', 0.0),
            'beta': self.config.get('beta', 0.0),
            'polynomial_type': self.config.get('polynomial_type', 'chebyshev'),
            'polynomial_params': {'alpha': self.config.get('alpha', 0.0), 'beta': self.config.get('beta', 0.0)}
        }
        
        return BipartiteSpecificFilter(
            self.n_users, self.n_items, self.filter_order,
            bipartite_init, True, bipartite_dim, bipartite_design, **filter_kwargs
        )
    
    def setup_combination_weights(self):
        """Setup fixed combination weights (not learnable)"""
        # Use fixed equal weights like the static model
        pass
    
    def forward(self, users):
        """Forward pass"""
        if users.device != self.adj_tensor.device:
            users = users.to(self.adj_tensor.device)
        
        batch_size = users.shape[0]
        user_profiles = self.adj_tensor[users]
        
        scores = [user_profiles]  # Direct collaborative filtering
        
        # Item-based filtering with multi-hop propagation
        if self.filter in ['i', 'ui', 'uib'] and hasattr(self, 'item_filter'):
            item_responses = self.item_filter(self.item_eigenvals, users)
            avg_item_response = item_responses.mean(dim=0)
            item_similarity_filtered = self.item_eigenvecs @ torch.diag(avg_item_response) @ self.item_eigenvecs.t()
            
            # Multi-hop item propagation
            if self.n_hops == 1:
                # 1-hop: user's items -> similar items
                item_filtered = user_profiles @ item_similarity_filtered
            elif self.n_hops == 2:
                # 2-hop: user's items -> similar items -> items similar to those
                hop1 = user_profiles @ item_similarity_filtered  # First hop
                hop2 = hop1 @ item_similarity_filtered  # Second hop through item similarities
                item_filtered = self.hop_weight * hop2 + (1 - self.hop_weight) * hop1
            else:  # 3-hop
                # 3-hop: extend further through item similarity network
                hop1 = user_profiles @ item_similarity_filtered
                hop2 = hop1 @ item_similarity_filtered  
                hop3 = hop2 @ item_similarity_filtered  # Third hop
                item_filtered = 0.5 * hop1 + 0.3 * hop2 + 0.2 * hop3
            
            scores.append(item_filtered)
        
        # User-based filtering with multi-hop propagation
        if self.filter in ['u', 'ui', 'uib', 'ub'] and hasattr(self, 'user_filter'):
            user_responses = self.user_filter(self.user_eigenvals, users)
            avg_user_response = user_responses.mean(dim=0)
            user_similarity_filtered = self.user_eigenvecs @ torch.diag(avg_user_response) @ self.user_eigenvecs.t()
            
            # Multi-hop user propagation
            if self.n_hops == 1:
                # 1-hop: user -> similar users -> their items
                user_filtered_batch = user_similarity_filtered[users] @ self.adj_tensor
            elif self.n_hops == 2:
                # 2-hop: user -> similar users -> users similar to those -> their items
                # First hop: find similar users
                similar_users_1hop = user_similarity_filtered[users]  # batch_size x n_users
                
                # Second hop: from those similar users, find their similar users
                similar_users_2hop = similar_users_1hop @ user_similarity_filtered  # batch_size x n_users
                
                # Get items from both 1-hop and 2-hop similar users
                items_1hop = similar_users_1hop @ self.adj_tensor
                items_2hop = similar_users_2hop @ self.adj_tensor
                
                user_filtered_batch = self.hop_weight * items_2hop + (1 - self.hop_weight) * items_1hop
            else:  # 3-hop
                # 3-hop: extend further through user similarity network
                similar_users_1hop = user_similarity_filtered[users]
                similar_users_2hop = similar_users_1hop @ user_similarity_filtered
                similar_users_3hop = similar_users_2hop @ user_similarity_filtered
                
                items_1hop = similar_users_1hop @ self.adj_tensor
                items_2hop = similar_users_2hop @ self.adj_tensor
                items_3hop = similar_users_3hop @ self.adj_tensor
                
                user_filtered_batch = 0.5 * items_1hop + 0.3 * items_2hop + 0.2 * items_3hop
            
            scores.append(user_filtered_batch)
        
        # Bipartite filtering with 2-hop propagation
        if self.filter in ['b', 'uib', 'ub'] and hasattr(self, 'bipartite_filter'):
            bipartite_responses = self.bipartite_filter(self.bipartite_eigenvals, users)
            avg_bipartite_response = bipartite_responses.mean(dim=0)
            
            # Apply spectral filter to bipartite adjacency matrix
            bipartite_filtered = self.bipartite_eigenvecs @ torch.diag(avg_bipartite_response) @ self.bipartite_eigenvecs.t()
            
            # Extract user-user and item-item parts from filtered bipartite matrix
            user_user_filtered = bipartite_filtered[:self.n_users, :self.n_users]
            item_item_filtered = bipartite_filtered[self.n_users:, self.n_users:]
            user_item_filtered = bipartite_filtered[:self.n_users, self.n_users:]
            
            # Store bipartite_filtered for use in all hop cases
            filtered_bipartite = bipartite_filtered
            
            # Multi-hop propagation based on n_hops setting
            if self.n_hops == 1:
                # 1-hop: Apply spectral filter directly to user profiles
                bipartite_batch = user_profiles @ item_item_filtered
            elif self.n_hops == 2:
                # 2-hop: user -> item -> user -> item with spectral filtering
                # Key insight: use the bipartite structure more effectively
                
                # Get the full bipartite adjacency structure
                bipartite_adj = torch.zeros(self.n_users + self.n_items, self.n_users + self.n_items, device=self.device)
                bipartite_adj[:self.n_users, self.n_users:] = self.adj_tensor
                bipartite_adj[self.n_users:, :self.n_users] = self.adj_tensor.t()
                
                # Apply spectral filter to bipartite adjacency
                filtered_bipartite = bipartite_filtered
                
                # Extract filtered user-item interactions
                filtered_ui = filtered_bipartite[:self.n_users, self.n_users:]
                filtered_iu = filtered_bipartite[self.n_users:, :self.n_users]
                
                # 2-hop: user -> item (filtered) -> user (filtered) -> item
                # This properly captures 2-hop collaborative signals
                user_to_user_2hop = filtered_ui @ filtered_iu  # Users who share items
                final_scores = user_to_user_2hop[users] @ self.adj_tensor
                
                # Combine with direct signal
                bipartite_batch = self.hop_weight * final_scores + (1 - self.hop_weight) * user_profiles
            else:
                # 3-hop propagation - corrected version
                # Extract components from filtered bipartite matrix
                filtered_ui = filtered_bipartite[:self.n_users, self.n_users:]  # U x I
                filtered_iu = filtered_bipartite[self.n_users:, :self.n_users]  # I x U
                
                # 3-hop: user -> item -> user -> item -> user -> item
                # Simplify by using item-item similarity through users
                
                # Compute item-item similarity through shared users
                # This is more stable than chaining multiple hops
                item_sim_through_users = self.adj_tensor.t() @ self.adj_tensor  # I x I
                
                # Apply spectral filtering to this similarity
                filtered_item_sim = item_item_filtered
                
                # For 3-hop, apply the filtered similarity multiple times
                # 1-hop: direct user profiles
                hop1 = user_profiles
                
                # 2-hop: through item similarities
                hop2 = hop1 @ filtered_item_sim
                
                # 3-hop: another round through item similarities
                hop3 = hop2 @ filtered_item_sim
                
                # Weighted combination emphasizing closer hops
                bipartite_batch = (0.5 * hop1 + 
                                 0.3 * hop2 + 
                                 0.2 * hop3)
            
            scores.append(bipartite_batch)
        
        # Combine predictions with fixed equal weights
        n_views = len(scores)
        predicted = sum(scores) / n_views
        
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


class UserSpecificFilter(nn.Module):
    """User-specific filter using complete filter collection"""
    
    def __init__(self, n_users, n_items, filter_order=6, init_filter_name='smooth', 
                 shared_base=True, personalization_dim=16, filter_design='enhanced_basis', **kwargs):
        super().__init__()
        self.n_users = n_users
        self.filter_order = filter_order
        self.personalization_dim = personalization_dim
        self.filter_design = filter_design
        
        # Create the actual filter using the factory function
        self.base_filter = fl.create_filter(
            filter_design, filter_order, init_filter_name, **kwargs
        )
        
        # Personalization layers
        self.user_embeddings = nn.Embedding(n_users, personalization_dim)
        self.adaptation_layer = nn.Linear(personalization_dim, filter_order + 1)
        self.adaptation_scale = nn.Parameter(torch.tensor(0.12))
        
        nn.init.normal_(self.user_embeddings.weight, 0, 0.008)
        nn.init.xavier_uniform_(self.adaptation_layer.weight)
        
        print(f"🔧 UserFilter: {filter_design} ({init_filter_name})")
    
    def forward(self, eigenvalues, user_ids):
        device = eigenvalues.device
        
        # Get base filter response
        base_response = self.base_filter(eigenvalues)
        
        # Get user-specific adaptations
        user_embeds = self.user_embeddings(user_ids)
        adaptations = self.adaptation_layer(user_embeds)
        adaptations = torch.tanh(adaptations) * self.adaptation_scale
        
        # Apply adaptations to the base response
        # This creates user-specific variations of the filter
        batch_size = user_ids.shape[0]
        eigenval_size = eigenvalues.shape[0]
        
        # Expand base response for batch processing
        if len(base_response.shape) == 1:
            expanded_base = base_response.unsqueeze(0).expand(batch_size, -1)
        else:
            expanded_base = base_response
        
        # Apply polynomial adaptations
        return self.apply_polynomial_adaptation(eigenvalues, adaptations, expanded_base)
    
    def apply_polynomial_adaptation(self, eigenvalues, user_coeffs, base_response):
        """Apply user-specific polynomial adaptations"""
        device = eigenvalues.device
        batch_size, n_coeffs = user_coeffs.shape
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        # Start with base response
        result = base_response
        
        # Add polynomial corrections
        correction = user_coeffs[:, 0:1]
        if n_coeffs >= 2:
            correction = correction + user_coeffs[:, 1:2] * x.unsqueeze(0)
        
        if n_coeffs >= 3:
            T_prev = torch.ones_like(x).unsqueeze(0)
            T_curr = x.unsqueeze(0)
            
            for i in range(2, n_coeffs):
                T_next = 2 * x.unsqueeze(0) * T_curr - T_prev
                correction = correction + user_coeffs[:, i:i+1] * T_next
                T_prev, T_curr = T_curr, T_next
        
        # Apply correction with clamping
        corrected_result = result + 0.1 * torch.tanh(correction)
        return torch.clamp(corrected_result, min=1e-6, max=2.0)


class ItemSpecificFilter(nn.Module):
    """Item-specific filter using complete filter collection"""
    
    def __init__(self, n_users, n_items, filter_order=6, init_filter_name='sharp',
                 shared_base=True, personalization_dim=12, filter_design='chebyshev', **kwargs):
        super().__init__()
        self.n_users = n_users
        self.filter_order = filter_order
        self.personalization_dim = personalization_dim
        self.filter_design = filter_design
        
        # Create the actual filter using the factory function
        self.base_filter = fl.create_filter(
            filter_design, filter_order, init_filter_name, **kwargs
        )
        
        # Personalization layers
        self.user_embeddings = nn.Embedding(n_users, personalization_dim)
        self.adaptation_layer = nn.Linear(personalization_dim, filter_order + 1)
        self.adaptation_scale = nn.Parameter(torch.tensor(0.18))
        
        nn.init.normal_(self.user_embeddings.weight, 0, 0.01)
        nn.init.xavier_uniform_(self.adaptation_layer.weight)
        
        print(f"🔧 ItemFilter: {filter_design} ({init_filter_name})")
    
    def forward(self, eigenvalues, user_ids):
        device = eigenvalues.device
        
        # Get base filter response
        base_response = self.base_filter(eigenvalues)
        
        # Get user-specific adaptations
        user_embeds = self.user_embeddings(user_ids)
        adaptations = self.adaptation_layer(user_embeds)
        adaptations = torch.tanh(adaptations) * self.adaptation_scale
        
        # Apply adaptations to the base response
        batch_size = user_ids.shape[0]
        
        # Expand base response for batch processing
        if len(base_response.shape) == 1:
            expanded_base = base_response.unsqueeze(0).expand(batch_size, -1)
        else:
            expanded_base = base_response
        
        # Apply polynomial adaptations
        return self.apply_polynomial_adaptation(eigenvalues, adaptations, expanded_base)
    
    def apply_polynomial_adaptation(self, eigenvalues, user_coeffs, base_response):
        """Apply user-specific polynomial adaptations"""
        device = eigenvalues.device
        batch_size, n_coeffs = user_coeffs.shape
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        # Start with base response
        result = base_response
        
        # Add polynomial corrections
        correction = user_coeffs[:, 0:1]
        if n_coeffs >= 2:
            correction = correction + user_coeffs[:, 1:2] * x.unsqueeze(0)
        
        if n_coeffs >= 3:
            T_prev = torch.ones_like(x).unsqueeze(0)
            T_curr = x.unsqueeze(0)
            
            for i in range(2, n_coeffs):
                T_next = 2 * x.unsqueeze(0) * T_curr - T_prev
                correction = correction + user_coeffs[:, i:i+1] * T_next
                T_prev, T_curr = T_curr, T_next
        
        # Apply correction with clamping
        corrected_result = result + 0.15 * torch.tanh(correction)
        return torch.clamp(corrected_result, min=1e-6, max=2.0)


class BipartiteSpecificFilter(nn.Module):
    """Bipartite-specific filter using complete filter collection"""
    
    def __init__(self, n_users, n_items, filter_order=6, init_filter_name='smooth',
                 shared_base=True, personalization_dim=20, filter_design='original', **kwargs):
        super().__init__()
        self.n_users = n_users
        self.filter_order = filter_order
        self.personalization_dim = personalization_dim
        self.filter_design = filter_design
        
        # Create the actual filter using the factory function
        self.base_filter = fl.create_filter(
            filter_design, filter_order, init_filter_name, **kwargs
        )
        
        # Personalization layers
        self.user_embeddings = nn.Embedding(n_users, personalization_dim)
        self.adaptation_layer = nn.Linear(personalization_dim, filter_order + 1)
        self.adaptation_scale = nn.Parameter(torch.tensor(0.22))
        self.user_item_interaction_weight = nn.Parameter(torch.tensor(1.0))
        
        nn.init.normal_(self.user_embeddings.weight, 0, 0.012)
        nn.init.xavier_uniform_(self.adaptation_layer.weight)
        
        print(f"🔧 BipartiteFilter: {filter_design} ({init_filter_name})")
    
    def forward(self, eigenvalues, user_ids):
        device = eigenvalues.device
        
        # Get base filter response
        base_response = self.base_filter(eigenvalues)
        
        # Get user-specific adaptations
        user_embeds = self.user_embeddings(user_ids)
        adaptations = self.adaptation_layer(user_embeds)
        adaptations = torch.tanh(adaptations) * self.adaptation_scale
        
        # Apply adaptations to the base response
        batch_size = user_ids.shape[0]
        
        # Expand base response for batch processing
        if len(base_response.shape) == 1:
            expanded_base = base_response.unsqueeze(0).expand(batch_size, -1)
        else:
            expanded_base = base_response
        
        # Apply polynomial adaptations
        return self.apply_polynomial_adaptation(eigenvalues, adaptations, expanded_base)
    
    def apply_polynomial_adaptation(self, eigenvalues, user_coeffs, base_response):
        """Apply user-specific polynomial adaptations"""
        device = eigenvalues.device
        batch_size, n_coeffs = user_coeffs.shape
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        # Start with base response
        result = base_response
        
        # Add polynomial corrections
        correction = user_coeffs[:, 0:1]
        if n_coeffs >= 2:
            correction = correction + user_coeffs[:, 1:2] * x.unsqueeze(0)
        
        if n_coeffs >= 3:
            T_prev = torch.ones_like(x).unsqueeze(0)
            T_curr = x.unsqueeze(0)
            
            for i in range(2, n_coeffs):
                T_next = 2 * x.unsqueeze(0) * T_curr - T_prev
                correction = correction + user_coeffs[:, i:i+1] * T_next
                T_prev, T_curr = T_curr, T_next
        
        # Apply correction with clamping and interaction weight
        interaction_weight = torch.sigmoid(self.user_item_interaction_weight)
        corrected_result = result + interaction_weight * 0.2 * torch.tanh(correction)
        return torch.clamp(corrected_result, min=1e-6, max=2.0)


# Simple configuration functions
def get_fast_config():
    return {
        'user_filter_design': 'enhanced_basis',
        'item_filter_design': 'chebyshev',
        'bipartite_filter_design': 'original',
        'user_personalization_dim': 8,
        'item_personalization_dim': 6,
        'bipartite_personalization_dim': 12,
        'u_n_eigen': 16,
        'i_n_eigen': 20,
        'b_n_eigen': 18
    }

def get_standard_config():
    return {
        'user_filter_design': 'enhanced_basis',
        'item_filter_design': 'chebyshev',
        'bipartite_filter_design': 'original',
        'user_personalization_dim': 16,
        'item_personalization_dim': 12,
        'bipartite_personalization_dim': 20
    }