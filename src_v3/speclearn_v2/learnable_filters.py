"""
Learnable Spectral Filters
Clean implementation with Bernstein, Chebyshev, and Spectral Basis
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.special import comb


class BernsteinFilter(nn.Module):
    """Bernstein polynomial filter - universal approximator on [0,1]"""
    
    def __init__(self, order=8, init_type='original'):
        super().__init__()
        self.order = order
        self.coefficients = nn.Parameter(torch.ones(order))
        
        # Initialize based on known good patterns
        if init_type == 'original':
            # Approximate (1-x)^n pattern
            with torch.no_grad():
                for i in range(order):
                    self.coefficients[i] = (1 - i/order)**2
        elif init_type == 'sharp':
            # Sharp cutoff at low frequencies
            with torch.no_grad():
                self.coefficients[:order//3] = 1.0
                self.coefficients[order//3:] = 0.1
        elif init_type == 'smooth':
            # Smooth decay
            with torch.no_grad():
                for i in range(order):
                    self.coefficients[i] = np.exp(-2 * i/order)
        elif init_type == 'linear_dec':
            # Linear decreasing - optimal for Yelp2018 item view
            with torch.no_grad():
                self.coefficients.data = torch.linspace(1.0, 0.1, order)
        elif init_type == 'step_0.5':
            # Step function at 0.5 threshold - optimal for Amazon-book item view
            with torch.no_grad():
                self.coefficients[:order//2] = 1.0
                self.coefficients[order//2:] = 0.1
        elif init_type == 'step_0.7':
            # Step function at 0.7 threshold - optimal for user views
            with torch.no_grad():
                cutoff = int(0.7 * order)
                self.coefficients[:cutoff] = 1.0
                self.coefficients[cutoff:] = 0.1
        elif init_type == 'step_0.9':
            # Step function at 0.9 threshold - optimal for Gowalla user view
            with torch.no_grad():
                cutoff = int(0.9 * order)
                self.coefficients[:cutoff] = 1.0
                self.coefficients[cutoff:] = 0.1
        elif init_type == 'exp_decay':
            # Exponential decay - optimal for Gowalla item view
            with torch.no_grad():
                for i in range(order):
                    self.coefficients[i] = np.exp(-5 * i / order)
        elif init_type == 'constant_1':
            # Constant 1 - optimal for Yelp2018 bipartite view
            with torch.no_grad():
                self.coefficients.data = torch.ones(order)
        elif init_type == 'constant_0.1':
            # Constant 0.1 - optimal for Gowalla bipartite view
            with torch.no_grad():
                self.coefficients.data = torch.ones(order) * 0.1
    
    def forward(self, eigenvals):
        # Ensure eigenvals are in [0,1] - they should be for normalized Laplacian
        eigenvals = torch.clamp(eigenvals, 0, 1)
        
        n = self.order - 1
        result = torch.zeros_like(eigenvals)
        
        # Use softmax to ensure coefficients sum to reasonable values
        coeffs = torch.softmax(self.coefficients, dim=0) * self.order
        
        for k in range(self.order):
            # Bernstein basis polynomial
            bernstein_basis = (comb(n, k) * 
                             torch.pow(eigenvals, k) * 
                             torch.pow(1 - eigenvals, n - k))
            result += coeffs[k] * bernstein_basis
            
        return result


class ChebyshevFilter(nn.Module):
    """Chebyshev polynomial filter - optimal minimax approximation"""
    
    def __init__(self, order=8, init_type='original'):
        super().__init__()
        self.order = order
        self.coefficients = nn.Parameter(torch.zeros(order))
        
        # Initialize
        if init_type == 'original':
            # Approximate (1-x)^n using Chebyshev expansion
            with torch.no_grad():
                self.coefficients[0] = 1.0
                self.coefficients[1] = -2.0
                for i in range(2, order):
                    self.coefficients[i] = (-1)**(i) * 0.5 / i
        elif init_type == 'lowpass':
            with torch.no_grad():
                self.coefficients[0] = 1.0
                self.coefficients[1:] = -0.5
        elif init_type in ['linear_dec', 'step_0.5', 'step_0.7', 'step_0.9', 'exp_decay', 'constant_1', 'constant_0.1']:
            # For optimal patterns, use pattern-inspired Chebyshev coefficients
            with torch.no_grad():
                if init_type == 'linear_dec':
                    # Linear decay pattern
                    for i in range(order):
                        self.coefficients[i] = (1.0 - 0.9 * i / order) * ((-1)**i * 0.5 / (i + 1))
                elif init_type in ['step_0.5', 'step_0.7', 'step_0.9']:
                    # Step function patterns
                    threshold = float(init_type.split('_')[1])
                    cutoff = int(threshold * order)
                    self.coefficients[:cutoff] = 1.0 / cutoff
                    self.coefficients[cutoff:] = 0.1 / max(1, order - cutoff)
                elif init_type == 'exp_decay':
                    # Exponential decay
                    for i in range(order):
                        self.coefficients[i] = np.exp(-3 * i / order) * ((-1)**i * 0.5)
                elif init_type == 'constant_1':
                    self.coefficients[0] = 1.0
                    self.coefficients[1:] = 0.0
                elif init_type == 'constant_0.1':
                    self.coefficients[0] = 0.1
                    self.coefficients[1:] = 0.0
    
    def forward(self, eigenvals):
        # Map eigenvalues to [-1, 1] for Chebyshev domain
        x = 2 * eigenvals - 1
        
        # Compute Chebyshev polynomials recursively
        if self.order >= 1:
            T0 = torch.ones_like(x)
            result = self.coefficients[0] * T0
            
        if self.order >= 2:
            T1 = x
            result = result + self.coefficients[1] * T1
            
        # Recursion: T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
        T_prev2 = T0 if self.order >= 1 else None
        T_prev1 = T1 if self.order >= 2 else None
        
        for n in range(2, self.order):
            T_n = 2 * x * T_prev1 - T_prev2
            result = result + self.coefficients[n] * T_n
            T_prev2, T_prev1 = T_prev1, T_n
            
        # Map back to [0, 1] range
        return torch.sigmoid(result)


class SpectralBasisFilter(nn.Module):
    """Learnable spectral basis with Gaussian-like frequency bands"""
    
    def __init__(self, n_bands=4, init_type='uniform'):
        super().__init__()
        self.n_bands = n_bands
        
        # Learnable parameters
        self.centers = nn.Parameter(torch.zeros(n_bands))
        self.widths = nn.Parameter(torch.ones(n_bands))
        self.weights = nn.Parameter(torch.ones(n_bands))
        
        # Initialize
        if init_type == 'uniform':
            # Uniformly spaced centers
            with torch.no_grad():
                self.centers.data = torch.linspace(0, 1, n_bands)
                self.widths.data = torch.ones(n_bands) * 0.2
        elif init_type == 'lowfreq':
            # Concentrate on low frequencies
            with torch.no_grad():
                self.centers.data = torch.linspace(0, 0.5, n_bands)
                self.widths.data = torch.ones(n_bands) * 0.1
        elif init_type == 'linear_dec':
            # Linear decreasing pattern
            with torch.no_grad():
                self.centers.data = torch.linspace(0, 1, n_bands)
                self.weights.data = torch.linspace(1.0, 0.1, n_bands)
                self.widths.data = torch.ones(n_bands) * 0.15
        elif init_type in ['step_0.5', 'step_0.7', 'step_0.9']:
            # Step function patterns
            threshold = float(init_type.split('_')[1])
            with torch.no_grad():
                self.centers.data = torch.linspace(0, 1, n_bands)
                self.weights.data = torch.where(self.centers.data < threshold, 1.0, 0.1)
                self.widths.data = torch.ones(n_bands) * 0.1
        elif init_type == 'exp_decay':
            # Exponential decay pattern
            with torch.no_grad():
                self.centers.data = torch.linspace(0, 1, n_bands)
                self.weights.data = torch.exp(-5 * self.centers.data)
                self.widths.data = torch.ones(n_bands) * 0.2
        elif init_type == 'constant_1':
            # Constant 1 pattern
            with torch.no_grad():
                self.centers.data = torch.linspace(0, 1, n_bands)
                self.weights.data = torch.ones(n_bands)
                self.widths.data = torch.ones(n_bands) * 0.3
        elif init_type == 'constant_0.1':
            # Constant 0.1 pattern
            with torch.no_grad():
                self.centers.data = torch.linspace(0, 1, n_bands)
                self.weights.data = torch.ones(n_bands) * 0.1
                self.widths.data = torch.ones(n_bands) * 0.3
    
    def forward(self, eigenvals):
        result = torch.zeros_like(eigenvals)
        
        # Ensure centers stay in [0,1]
        centers = torch.sigmoid(self.centers)
        # Ensure widths are positive
        widths = torch.nn.functional.softplus(self.widths) + 0.01
        
        for i in range(self.n_bands):
            # Gaussian basis function
            basis = torch.exp(-((eigenvals - centers[i])**2) / (2 * widths[i]**2))
            result += self.weights[i] * basis
            
        return torch.nn.functional.softplus(result)  # Ensure positive output


class LearnableSpectralFilter(nn.Module):
    """Wrapper for different filter types"""
    
    def __init__(self, filter_type='bernstein', order=8, init_type='original'):
        super().__init__()
        self.filter_type = filter_type
        
        if filter_type == 'bernstein':
            self.filter = BernsteinFilter(order, init_type)
        elif filter_type == 'chebyshev':
            self.filter = ChebyshevFilter(order, init_type)
        elif filter_type == 'spectral_basis':
            n_bands = max(4, order // 2)
            self.filter = SpectralBasisFilter(n_bands, init_type)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
    
    def forward(self, eigenvals):
        return self.filter(eigenvals)
    
    def get_filter_values(self, n_points=100):
        """Get filter response for visualization"""
        with torch.no_grad():
            x = torch.linspace(0, 1, n_points)
            y = self.forward(x)
            return x.numpy(), y.numpy()