"""
Full Spectrum Spectral CF - Enhanced version with full spectrum access
Explores different ways to access the full spectrum while maintaining efficiency
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, svds
import time


class FullSpectrumFilter(nn.Module):
    """Enhanced filter with full spectrum access options"""
    def __init__(self, n_eigen, mode='truncated', full_size=None):
        super().__init__()
        self.n_eigen = n_eigen
        self.mode = mode
        self.full_size = full_size or n_eigen
        
        if mode == 'truncated':
            # Standard truncated approach (like our current)
            self.filter_weights = nn.Parameter(torch.ones(n_eigen))
        elif mode == 'extended':
            # Extended: learn weights for truncated + extrapolate to full
            self.filter_weights = nn.Parameter(torch.ones(n_eigen))
            self.extrapolation_param = nn.Parameter(torch.tensor(0.1))
        elif mode == 'polynomial':
            # Polynomial approximation for full spectrum
            self.poly_order = min(8, n_eigen // 4)
            self.poly_coeffs = nn.Parameter(torch.zeros(self.poly_order + 1))
            # Initialize polynomial for low-pass
            with torch.no_grad():
                self.poly_coeffs[0] = 1.0
                for i in range(1, self.poly_order + 1):
                    self.poly_coeffs[i] = 0.3 * ((-1)**i) / (i + 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        with torch.no_grad():
            if hasattr(self, 'filter_weights'):
                # Low-pass initialization
                for i in range(self.n_eigen):
                    self.filter_weights[i] = 0.8 * np.exp(-2 * i / self.n_eigen)
    
    def forward(self, eigenvals):
        """Apply filter based on mode"""
        if self.mode == 'truncated':
            return torch.sigmoid(self.filter_weights)
        
        elif self.mode == 'extended':
            # Use learned weights + extrapolate for missing spectrum
            base_response = torch.sigmoid(self.filter_weights)
            if len(eigenvals) > self.n_eigen:
                # Extrapolate for higher frequencies
                extra_len = len(eigenvals) - self.n_eigen
                extra_response = torch.sigmoid(self.extrapolation_param) * torch.exp(-torch.arange(extra_len, dtype=torch.float32, device=eigenvals.device))
                return torch.cat([base_response, extra_response])
            return base_response
        
        elif self.mode == 'polynomial':
            # Polynomial approximation to capture full spectrum behavior
            # Normalize eigenvals to [0,1] for stability
            normalized_eigs = (eigenvals - eigenvals.min()) / (eigenvals.max() - eigenvals.min() + 1e-8)
            
            # Evaluate polynomial: sum(c_k * x^k)
            result = self.poly_coeffs[0] * torch.ones_like(normalized_eigs)
            x_power = normalized_eigs
            for k in range(1, self.poly_order + 1):
                result = result + self.poly_coeffs[k] * x_power
                x_power = x_power * normalized_eigs
            
            return torch.sigmoid(result)


class FullSpectrumCF(nn.Module):
    """Full spectrum spectral CF with multiple access modes"""
    
    def __init__(self, adj_mat, config):
        super().__init__()
        self.device = config.get('device', torch.device('cpu'))
        
        # Convert adjacency matrix
        if sp.issparse(adj_mat):
            self.adj_mat = adj_mat.tocsr()
        else:
            self.adj_mat = sp.csr_matrix(adj_mat)
        
        self.n_users, self.n_items = self.adj_mat.shape
        
        # Spectrum access configuration
        self.spectrum_mode = config.get('spectrum_mode', 'truncated')  # 'truncated', 'extended', 'polynomial', 'svd'
        
        # Eigenvalue configuration
        self.u_eigen = config.get('u_eigen', 64)
        self.i_eigen = config.get('i_eigen', 256) 
        self.b_eigen = config.get('b_eigen', 256)
        
        # For SVD mode, use more components
        if self.spectrum_mode == 'svd':
            self.svd_components = config.get('svd_components', 512)
        
        print(f"Full Spectrum CF: {self.n_users} users, {self.n_items} items")
        print(f"Spectrum access mode: {self.spectrum_mode}")
        print(f"Eigenvalues: u={self.u_eigen}, i={self.i_eigen}, b={self.b_eigen}")
        
        # Create filters based on mode
        if self.spectrum_mode == 'svd':
            self._setup_svd_decomposition()
        else:
            # Create learnable filters
            self.user_filter = FullSpectrumFilter(self.u_eigen, self.spectrum_mode, self.n_users)
            self.item_filter = FullSpectrumFilter(self.i_eigen, self.spectrum_mode, self.n_items) 
            self.bipartite_filter = FullSpectrumFilter(self.b_eigen, self.spectrum_mode, self.n_users + self.n_items)
            
            # Compute eigendecompositions
            self._setup_eigendecompositions()
        
        # Learning rates per view
        self.user_lr = config.get('user_lr', 0.01)
        self.item_lr = config.get('item_lr', 0.01)
        self.bipartite_lr = config.get('bipartite_lr', 0.01)
    
    def _setup_svd_decomposition(self):
        """SVD-based approach (like ChebyCF) for full spectrum access"""
        print("Setting up SVD decomposition for full spectrum access...")
        start = time.time()
        
        # GF-CF style normalization
        rowsum = np.array(self.adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt_u = np.power(rowsum + 1e-10, -0.5)
        d_inv_sqrt_u[np.isinf(d_inv_sqrt_u)] = 0.
        d_mat_u = sp.diags(d_inv_sqrt_u)
        
        colsum = np.array(self.adj_mat.sum(axis=0)).flatten()
        d_inv_sqrt_i = np.power(colsum + 1e-10, -0.5)
        d_inv_sqrt_i[np.isinf(d_inv_sqrt_i)] = 0.
        d_mat_i = sp.diags(d_inv_sqrt_i)
        
        # Normalized adjacency
        norm_adj = d_mat_u @ self.adj_mat @ d_mat_i
        
        # SVD of normalized adjacency (like ChebyCF)
        print(f"Computing SVD with {self.svd_components} components...")
        U, s, Vt = svds(norm_adj, k=min(self.svd_components, min(norm_adj.shape) - 1), which='LM')
        
        self.register_buffer('svd_U', torch.tensor(U, dtype=torch.float32))
        self.register_buffer('svd_s', torch.tensor(s, dtype=torch.float32))
        self.register_buffer('svd_Vt', torch.tensor(Vt, dtype=torch.float32))
        
        # Learnable coefficients for SVD filtering
        self.svd_filter = nn.Parameter(torch.ones(len(s)))
        with torch.no_grad():
            # Initialize as low-pass
            for i in range(len(s)):
                self.svd_filter[i] = 0.8 * np.exp(-2 * i / len(s))
        
        print(f"SVD decomposition completed in {time.time() - start:.2f}s")
    
    def _setup_eigendecompositions(self):
        """Standard eigendecomposition setup"""
        start = time.time()
        print("Computing eigendecompositions...")
        
        # GF-CF style normalization
        rowsum = np.array(self.adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt_u = np.power(rowsum + 1e-10, -0.5)
        d_inv_sqrt_u[np.isinf(d_inv_sqrt_u)] = 0.
        d_mat_u = sp.diags(d_inv_sqrt_u)
        
        colsum = np.array(self.adj_mat.sum(axis=0)).flatten()
        d_inv_sqrt_i = np.power(colsum + 1e-10, -0.5)
        d_inv_sqrt_i[np.isinf(d_inv_sqrt_i)] = 0.
        d_mat_i = sp.diags(d_inv_sqrt_i)
        
        # Normalized adjacency
        norm_adj = d_mat_u @ self.adj_mat @ d_mat_i
        
        # Compute eigendecompositions
        user_sim = norm_adj @ norm_adj.T
        u_vals, u_vecs = eigsh(user_sim, k=min(self.u_eigen, user_sim.shape[0]-1), which='LM')
        self.register_buffer('user_eigenvals', torch.tensor(u_vals, dtype=torch.float32))
        self.register_buffer('user_eigenvecs', torch.tensor(u_vecs, dtype=torch.float32))
        
        item_sim = norm_adj.T @ norm_adj
        i_vals, i_vecs = eigsh(item_sim, k=min(self.i_eigen, item_sim.shape[0]-1), which='LM')
        self.register_buffer('item_eigenvals', torch.tensor(i_vals, dtype=torch.float32))
        self.register_buffer('item_eigenvecs', torch.tensor(i_vecs, dtype=torch.float32))
        
        bipartite_adj = sp.bmat([[None, norm_adj], [norm_adj.T, None]], format='csr')
        b_vals, b_vecs = eigsh(bipartite_adj, k=min(self.b_eigen, bipartite_adj.shape[0]-1), which='LM')
        self.register_buffer('bipartite_eigenvals', torch.tensor(b_vals, dtype=torch.float32))
        self.register_buffer('bipartite_eigenvecs', torch.tensor(b_vecs, dtype=torch.float32))
        
        print(f"Eigendecomposition completed in {time.time() - start:.2f}s")
        
    def forward(self, users):
        """Forward pass with full spectrum access"""
        if not isinstance(users, torch.Tensor):
            users = torch.tensor(users, dtype=torch.long).to(self.device)
        
        # Get user interaction profiles
        batch_size = len(users)
        user_profiles = torch.zeros(batch_size, self.n_items, device=self.device)
        for i, user_id in enumerate(users):
            items = self.adj_mat[user_id.item()].indices
            if len(items) > 0:
                user_profiles[i, items] = 1.0
        
        if self.spectrum_mode == 'svd':
            # SVD-based filtering (full spectrum access)
            filter_response = torch.sigmoid(self.svd_filter)
            
            # Apply filtering: U @ diag(filter * s) @ Vt
            filtered_s = filter_response * self.svd_s
            
            # For user profiles: profiles @ V @ diag(filtered_s) @ Vt
            # This approximates full spectrum filtering
            temp = user_profiles @ self.svd_Vt.T  # (batch, svd_components)
            temp = temp * filtered_s.unsqueeze(0)  # Apply filtered singular values
            final_scores = temp @ self.svd_Vt  # (batch, n_items)
            
            return final_scores
        else:
            # Standard eigendecomposition with enhanced filters
            scores = []
            
            # User view
            user_filter_response = self.user_filter(self.user_eigenvals)
            user_vecs = self.user_eigenvecs[users]
            user_filtered = user_vecs @ torch.diag(user_filter_response) @ user_vecs.T @ user_profiles
            scores.append(user_filtered)
            
            # Item view
            item_filter_response = self.item_filter(self.item_eigenvals)
            item_filtered = user_profiles @ self.item_eigenvecs @ torch.diag(item_filter_response) @ self.item_eigenvecs.T
            scores.append(item_filtered)
            
            # Bipartite view
            bipartite_filter_response = self.bipartite_filter(self.bipartite_eigenvals)
            bipartite_vecs = self.bipartite_eigenvecs[users]
            bipartite_filtered = bipartite_vecs @ torch.diag(bipartite_filter_response) @ bipartite_vecs.T @ user_profiles
            scores.append(bipartite_filtered)
            
            # Combine views
            final_scores = sum(scores) / len(scores)
            return final_scores
    
    def getUsersRating(self, batch_users):
        """Get ratings for evaluation"""
        self.eval()
        with torch.no_grad():
            scores = self.forward(batch_users)
            return scores.cpu().numpy()
    
    def get_optimizer_groups(self):
        """Per-view learning rates"""
        groups = []
        
        if self.spectrum_mode == 'svd':
            groups.append({
                'params': [self.svd_filter], 
                'lr': self.item_lr,  # Use item lr for SVD 
                'name': 'svd_filter'
            })
        else:
            groups.extend([
                {'params': self.user_filter.parameters(), 'lr': self.user_lr, 'name': 'user_filter'},
                {'params': self.item_filter.parameters(), 'lr': self.item_lr, 'name': 'item_filter'},
                {'params': self.bipartite_filter.parameters(), 'lr': self.bipartite_lr, 'name': 'bipartite_filter'}
            ])
        
        return groups