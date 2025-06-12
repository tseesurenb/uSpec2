'''
Created on June 12, 2025
Complete Universal Spectral Filter Collection - ALL FILTERS INCLUDED
Minimalist approach with comprehensive filter support

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import torch
import torch.nn as nn
import numpy as np
import math

# Essential filter patterns (complete set)
filter_patterns = {
    'smooth': [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015],
    'sharp': [0.3, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8],
    'bandpass': [0.1, 0.2, 0.8, 1.0, 0.8, 0.2, 0.1],
    'golden_036': [1.0, -0.36, 0.1296, -0.220, 0.1564, -0.088, 0.0548],
    'butterworth': [1.0, -0.6, 0.2, -0.05, 0.01, -0.002, 0.0003],
    'gaussian': [1.0, -0.7, 0.15, -0.03, 0.005, -0.0007, 0.00008],
    'band_stop': [1.0, -0.8, 0.3, -0.7, 0.4, -0.1, 0.05],
    'notch': [1.0, -0.2, -0.3, 0.6, -0.4, 0.15, -0.05],
}

def get_filter_coefficients(filter_name, order=None, as_tensor=False, device=None):
    """Get filter coefficients by name with proper device placement."""
    if filter_name not in filter_patterns:
        raise ValueError(f"Unknown filter pattern: {filter_name}. Available: {list(filter_patterns.keys())}")
    
    coeffs = filter_patterns[filter_name].copy()
    
    if order is not None:
        if len(coeffs) > order + 1:
            coeffs = coeffs[:order + 1]
        elif len(coeffs) < order + 1:
            coeffs.extend([0.0] * (order + 1 - len(coeffs)))
    
    if as_tensor:
        tensor = torch.tensor(coeffs, dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    
    return coeffs

# =============================================================================
# POLYNOMIAL BASIS CLASSES (COMPLETE SET)
# =============================================================================

class ChebyshevBasis:
    @staticmethod
    def compute_polynomials(x, order, **kwargs):
        polynomials = []
        if order >= 0:
            T_0 = torch.ones_like(x)
            polynomials.append(T_0)
        if order >= 1:
            T_1 = x.clone()
            polynomials.append(T_1)
        if order >= 2:
            T_prev, T_curr = T_0, T_1
            for n in range(2, order + 1):
                T_next = 2 * x * T_curr - T_prev
                polynomials.append(T_next)
                T_prev, T_curr = T_curr, T_next
        return torch.stack(polynomials, dim=-1)

class JacobiBasis:
    @staticmethod
    def compute_polynomials(x, order, alpha=0.0, beta=0.0, **kwargs):
        polynomials = []
        if order >= 0:
            P_0 = torch.ones_like(x)
            polynomials.append(P_0)
        if order >= 1:
            P_1 = 0.5 * ((alpha - beta) + (alpha + beta + 2) * x)
            polynomials.append(P_1)
        if order >= 2:
            P_prev, P_curr = P_0, P_1
            for n in range(2, order + 1):
                a1 = 2 * n * (n + alpha + beta) * (2 * n + alpha + beta - 2)
                a2 = (2 * n + alpha + beta - 1) * (alpha**2 - beta**2)
                a3 = (2 * n + alpha + beta - 1) * (2 * n + alpha + beta) * (2 * n + alpha + beta - 2)
                a4 = 2 * (n + alpha - 1) * (n + beta - 1) * (2 * n + alpha + beta)
                
                P_next = ((a2 + a3 * x) * P_curr - a4 * P_prev) / a1
                polynomials.append(P_next)
                P_prev, P_curr = P_curr, P_next
        return torch.stack(polynomials, dim=-1)

class LegendreBasis:
    @staticmethod
    def compute_polynomials(x, order, **kwargs):
        polynomials = []
        if order >= 0:
            P_0 = torch.ones_like(x)
            polynomials.append(P_0)
        if order >= 1:
            P_1 = x.clone()
            polynomials.append(P_1)
        if order >= 2:
            P_prev, P_curr = P_0, P_1
            for n in range(1, order):
                P_next = ((2 * n + 1) * x * P_curr - n * P_prev) / (n + 1)
                polynomials.append(P_next)
                P_prev, P_curr = P_curr, P_next
        return torch.stack(polynomials, dim=-1)

class LaguerreBasis:
    @staticmethod
    def compute_polynomials(x, order, alpha=0.0, **kwargs):
        polynomials = []
        if order >= 0:
            L_0 = torch.ones_like(x)
            polynomials.append(L_0)
        if order >= 1:
            L_1 = 1 + alpha - x
            polynomials.append(L_1)
        if order >= 2:
            L_prev, L_curr = L_0, L_1
            for n in range(1, order):
                L_next = ((2 * n + 1 + alpha - x) * L_curr - (n + alpha) * L_prev) / (n + 1)
                polynomials.append(L_next)
                L_prev, L_curr = L_curr, L_next
        return torch.stack(polynomials, dim=-1)

class HermiteBasis:
    @staticmethod
    def compute_polynomials(x, order, **kwargs):
        polynomials = []
        if order >= 0:
            H_0 = torch.ones_like(x)
            polynomials.append(H_0)
        if order >= 1:
            H_1 = 2 * x
            polynomials.append(H_1)
        if order >= 2:
            H_prev, H_curr = H_0, H_1
            for n in range(1, order):
                H_next = 2 * x * H_curr - 2 * n * H_prev
                polynomials.append(H_next)
                H_prev, H_curr = H_curr, H_next
        return torch.stack(polynomials, dim=-1)

class BernsteinBasis:
    @staticmethod
    def compute_polynomials(x, order, **kwargs):
        polynomials = []
        n = order
        for k in range(order + 1):
            binom_coef = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
            if k == 0:
                B_k = binom_coef * torch.pow(1 - x, n - k)
            elif k == n:
                B_k = binom_coef * torch.pow(x, k)
            else:
                B_k = binom_coef * torch.pow(x, k) * torch.pow(1 - x, n - k)
            polynomials.append(B_k)
        return torch.stack(polynomials, dim=-1)

# =============================================================================
# FILTER DESIGN 1: ORIGINAL UNIVERSAL FILTER
# =============================================================================
class UniversalSpectralFilter(nn.Module):
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        lowpass = get_filter_coefficients(init_filter_name, as_tensor=True)
        coeffs_data = torch.zeros(filter_order + 1)
        for i, val in enumerate(lowpass[:filter_order + 1]):
            coeffs_data[i] = val

        self.register_buffer('init_coeffs', coeffs_data.clone())
        self.coeffs = nn.Parameter(coeffs_data.clone())
    
    def forward(self, eigenvalues):
        coeffs = self.coeffs.to(eigenvalues.device)
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        result = coeffs[0] * torch.ones_like(x)
        
        if len(coeffs) > 1:
            T_prev, T_curr = torch.ones_like(x), x
            result += coeffs[1] * T_curr
            
            for i in range(2, len(coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        filter_response = torch.exp(-torch.abs(result).clamp(max=10.0)) + 1e-6
        return filter_response

# =============================================================================
# FILTER DESIGN 2: SPECTRAL BASIS FILTER
# =============================================================================
class SpectralBasisFilter(nn.Module):
    """Learnable combination of proven filter patterns"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        filter_names = ['golden_036', 'smooth', 'butterworth', 'gaussian']
        
        for i, name in enumerate(filter_names):
            coeffs = get_filter_coefficients(name, order=filter_order, as_tensor=True)
            if len(coeffs) < filter_order + 1:
                padded_coeffs = torch.zeros(filter_order + 1)
                padded_coeffs[:len(coeffs)] = coeffs
                coeffs = padded_coeffs
            elif len(coeffs) > filter_order + 1:
                coeffs = coeffs[:filter_order + 1]
            
            self.register_buffer(f'filter_{i}', coeffs)
        
        self.filter_names = filter_names
        
        init_weights = torch.ones(len(filter_names)) * 0.25
        if init_filter_name in filter_names:
            init_idx = filter_names.index(init_filter_name)
            init_weights[init_idx] = 0.4
        
        self.mixing_weights = nn.Parameter(init_weights)
        self.refinement_coeffs = nn.Parameter(torch.zeros(filter_order + 1))
        self.refinement_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        weights = torch.softmax(self.mixing_weights, dim=0).to(device)
        
        filter_bank = []
        for i in range(len(self.filter_names)):
            filter_tensor = getattr(self, f'filter_{i}').to(device)
            filter_bank.append(filter_tensor)
        
        mixed_coeffs = torch.zeros_like(filter_bank[0])
        for i, base_filter in enumerate(filter_bank):
            mixed_coeffs += weights[i] * base_filter
        
        refinement_coeffs = self.refinement_coeffs.to(device)
        refinement_scale = self.refinement_scale.to(device)
        final_coeffs = mixed_coeffs + refinement_scale * refinement_coeffs
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        result = final_coeffs[0] * torch.ones_like(x)
        if len(final_coeffs) > 1:
            T_prev, T_curr = torch.ones_like(x), x
            result += final_coeffs[1] * T_curr
            
            for i in range(2, len(final_coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += final_coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        return torch.exp(-torch.abs(result).clamp(max=10.0)) + 1e-6

# =============================================================================
# FILTER DESIGN 3: ENHANCED SPECTRAL BASIS FILTER
# =============================================================================
class EnhancedSpectralBasisFilter(nn.Module):
    """Enhanced basis filter for maximum performance"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        filter_names = ['golden_036', 'smooth', 'butterworth', 'gaussian']
        
        for i, name in enumerate(filter_names):
            try:
                coeffs = get_filter_coefficients(name, order=filter_order, as_tensor=True)
                if len(coeffs) < filter_order + 1:
                    padded_coeffs = torch.zeros(filter_order + 1)
                    padded_coeffs[:len(coeffs)] = coeffs
                    coeffs = padded_coeffs
                elif len(coeffs) > filter_order + 1:
                    coeffs = coeffs[:filter_order + 1]
                
                self.register_buffer(f'filter_{i}', coeffs)
            except:
                continue
        
        self.filter_names = filter_names[:4]
        
        init_weights = torch.ones(len(self.filter_names)) * 0.25
        
        for i, name in enumerate(self.filter_names):
            if name == init_filter_name:
                init_weights[i] = 0.4
            elif name == 'golden_036':
                init_weights[i] = 0.3
        
        init_weights = init_weights / init_weights.sum()
        
        self.mixing_weights = nn.Parameter(init_weights)
        self.refinement_coeffs = nn.Parameter(torch.zeros(filter_order + 1))
        self.refinement_scale = nn.Parameter(torch.tensor(0.2))
        self.transform_scale = nn.Parameter(torch.tensor(1.0))
        self.transform_bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        weights = torch.softmax(self.mixing_weights, dim=0).to(device)
        
        filter_bank = []
        for i in range(len(self.filter_names)):
            filter_tensor = getattr(self, f'filter_{i}').to(device)
            filter_bank.append(filter_tensor)
        
        mixed_coeffs = torch.zeros_like(filter_bank[0])
        for i, base_filter in enumerate(filter_bank):
            mixed_coeffs += weights[i] * base_filter
        
        refinement_coeffs = self.refinement_coeffs.to(device)
        refinement_scale = self.refinement_scale.to(device)
        transform_scale = self.transform_scale.to(device)
        transform_bias = self.transform_bias.to(device)
        
        refinement = refinement_scale * torch.tanh(refinement_coeffs)
        final_coeffs = mixed_coeffs + refinement
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        result = final_coeffs[0] * torch.ones_like(x)
        if len(final_coeffs) > 1:
            T_prev, T_curr = torch.ones_like(x), x
            result += final_coeffs[1] * T_curr
            
            for i in range(2, len(final_coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += final_coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        result = transform_scale * result + transform_bias
        
        return torch.exp(-torch.abs(result).clamp(max=10.0)) + 1e-6

# =============================================================================
# FILTER DESIGN 4: ADAPTIVE GOLDEN FILTER
# =============================================================================
class AdaptiveGoldenFilter(nn.Module):
    """Learns adaptive variations of golden ratio patterns"""
    
    def __init__(self, filter_order=6, init_filter_name='golden_036'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        base_coeffs = get_filter_coefficients('golden_036', as_tensor=True)
        if len(base_coeffs) < filter_order + 1:
            padded_coeffs = torch.zeros(filter_order + 1)
            padded_coeffs[:len(base_coeffs)] = base_coeffs
            base_coeffs = padded_coeffs
        elif len(base_coeffs) > filter_order + 1:
            base_coeffs = base_coeffs[:filter_order + 1]
        
        self.register_buffer('base_coeffs', base_coeffs.clone())
        
        self.scale_factors = nn.Parameter(torch.ones(filter_order + 1))
        self.bias_terms = nn.Parameter(torch.zeros(filter_order + 1) * 0.1)
        self.golden_ratio_delta = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        
        adaptive_ratio = 0.36 + 0.1 * torch.tanh(self.golden_ratio_delta.to(device))
        
        scale_factors = self.scale_factors.to(device)
        bias_terms = self.bias_terms.to(device)
        base_coeffs = self.base_coeffs.to(device)
        
        scale_constrained = 0.5 + 0.5 * torch.sigmoid(scale_factors)
        bias_constrained = 0.1 * torch.tanh(bias_terms)
        
        adapted_coeffs = scale_constrained * base_coeffs + bias_constrained
        adapted_coeffs = adapted_coeffs.clone()
        adapted_coeffs[1] = -adaptive_ratio
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        result = adapted_coeffs[0] * torch.ones_like(x)
        if len(adapted_coeffs) > 1:
            T_prev, T_curr = torch.ones_like(x), x
            result += adapted_coeffs[1] * T_curr
            
            for i in range(2, len(adapted_coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += adapted_coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        return torch.exp(-torch.abs(result).clamp(max=10.0)) + 1e-6

# =============================================================================
# FILTER DESIGN 5: MULTI-SCALE SPECTRAL FILTER
# =============================================================================
class MultiScaleSpectralFilter(nn.Module):
    """Multi-scale spectral filtering with learnable frequency bands"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth', n_bands=4):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        self.n_bands = n_bands
        
        init_boundaries = torch.linspace(0, 1, n_bands + 1)[1:-1]
        self.band_boundaries = nn.Parameter(init_boundaries)
        self.band_responses = nn.Parameter(torch.ones(n_bands) * 0.5)
        self.transition_sharpness = nn.Parameter(torch.tensor(10.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval
        
        band_boundaries = self.band_boundaries.to(device)
        band_responses = self.band_responses.to(device)
        transition_sharpness = self.transition_sharpness.to(device)
        
        sorted_boundaries = torch.sort(band_boundaries)[0]
        boundaries = torch.cat([torch.zeros(1, device=device), 
                               sorted_boundaries,
                               torch.ones(1, device=device)])
        
        sharpness = torch.abs(transition_sharpness) + 1.0
        band_responses_sigmoid = torch.sigmoid(band_responses)
        
        response = torch.zeros_like(norm_eigenvals)
        
        for i in range(self.n_bands):
            left_boundary = boundaries[i]
            right_boundary = boundaries[i + 1]
            
            left_transition = torch.sigmoid(sharpness * (norm_eigenvals - left_boundary))
            right_transition = torch.sigmoid(sharpness * (right_boundary - norm_eigenvals))
            
            band_membership = left_transition * right_transition
            response += band_membership * band_responses_sigmoid[i]
        
        return torch.clamp(response, min=1e-6, max=1.0)

# =============================================================================
# FILTER DESIGN 6: BAND-STOP SPECTRAL FILTER
# =============================================================================
class BandStopSpectralFilter(nn.Module):
    """Band-stop filter that passes low and high frequencies, rejects middle"""
    
    def __init__(self, filter_order=6, init_filter_name='band_stop'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        self.low_cutoff = nn.Parameter(torch.tensor(0.2))
        self.high_cutoff = nn.Parameter(torch.tensor(1.0))
        self.low_gain = nn.Parameter(torch.tensor(1.0))
        self.high_gain = nn.Parameter(torch.tensor(0.8))
        self.stop_depth = nn.Parameter(torch.tensor(0.05))
        self.transition_sharpness = nn.Parameter(torch.tensor(5.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        
        low_cutoff = self.low_cutoff.to(device)
        high_cutoff = self.high_cutoff.to(device)
        low_gain = self.low_gain.to(device)
        high_gain = self.high_gain.to(device)
        stop_depth = self.stop_depth.to(device)
        transition_sharpness = self.transition_sharpness.to(device)
        
        low_cut = torch.sigmoid(low_cutoff) * 0.8
        high_cut = low_cut + torch.sigmoid(high_cutoff) * (2.0 - low_cut)
        low_g = torch.sigmoid(low_gain)
        high_g = torch.sigmoid(high_gain)
        stop_d = torch.sigmoid(stop_depth) * 0.2
        sharpness = torch.abs(transition_sharpness) + 1.0
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval
        norm_low_cut = low_cut / 2.0
        norm_high_cut = high_cut / 2.0
        
        response = torch.ones_like(norm_eigenvals)
        
        low_mask = norm_eigenvals < norm_low_cut
        response = torch.where(low_mask, low_g * response, response)
        
        high_mask = norm_eigenvals > norm_high_cut
        response = torch.where(high_mask, high_g * response, response)
        
        stop_mask = (norm_eigenvals >= norm_low_cut) & (norm_eigenvals <= norm_high_cut)
        if stop_mask.any():
            stop_center = (norm_low_cut + norm_high_cut) / 2
            stop_width = (norm_high_cut - norm_low_cut) / 2
            stop_response = stop_d + (1 - stop_d) * torch.exp(-sharpness * ((norm_eigenvals - stop_center) / stop_width) ** 2)
            response = torch.where(stop_mask, stop_response, response)
        
        return torch.clamp(response, min=1e-6, max=1.0)

# =============================================================================
# FILTER DESIGN 7: ADAPTIVE BAND-STOP FILTER
# =============================================================================
class AdaptiveBandStopFilter(nn.Module):
    """Advanced band-stop with multiple learnable stop bands"""
    
    def __init__(self, filter_order=6, init_filter_name='band_stop', n_stop_bands=2):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        self.n_stop_bands = n_stop_bands
        
        self.band_centers = nn.Parameter(torch.linspace(0.3, 1.2, n_stop_bands))
        self.band_widths = nn.Parameter(torch.ones(n_stop_bands) * 0.2)
        self.band_depths = nn.Parameter(torch.ones(n_stop_bands) * 0.1)
        
        self.base_gain = nn.Parameter(torch.tensor(1.0))
        self.high_freq_boost = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval * 2
        
        band_centers = self.band_centers.to(device)
        band_widths = self.band_widths.to(device)
        band_depths = self.band_depths.to(device)
        base_gain = self.base_gain.to(device)
        high_freq_boost = self.high_freq_boost.to(device)
        
        base_g = torch.sigmoid(base_gain)
        response = base_g * torch.ones_like(norm_eigenvals)
        
        for i in range(self.n_stop_bands):
            center = torch.clamp(band_centers[i], 0.1, 1.9)
            width = torch.sigmoid(band_widths[i]) * 0.5 + 0.05
            depth = torch.sigmoid(band_depths[i]) * 0.8
            
            band_attenuation = 1 - depth * torch.exp(-((norm_eigenvals - center) / width) ** 2)
            response *= band_attenuation
        
        high_boost = torch.sigmoid(high_freq_boost) * 0.5
        high_freq_mask = norm_eigenvals > 1.5
        response = torch.where(high_freq_mask, response + high_boost, response)
        
        return torch.clamp(response, min=1e-6, max=1.5)

# =============================================================================
# FILTER DESIGN 8: PARAMETRIC MULTI-BAND FILTER
# =============================================================================
class ParametricMultiBandFilter(nn.Module):
    """Parametric filter with learnable frequency bands (low/mid/high gains)"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth', n_bands=4):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        self.n_bands = n_bands
        
        init_boundaries = torch.linspace(0.2, 1.8, n_bands - 1)
        self.band_boundaries = nn.Parameter(init_boundaries)
        
        self.band_gains = nn.Parameter(torch.ones(n_bands))
        
        self.transition_sharpness = nn.Parameter(torch.tensor(3.0))
        self.global_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval * 2
        
        band_boundaries = self.band_boundaries.to(device)
        band_gains = self.band_gains.to(device)
        transition_sharpness = self.transition_sharpness.to(device)
        global_scale = self.global_scale.to(device)
        
        sorted_boundaries = torch.sort(band_boundaries)[0]
        boundaries = torch.cat([torch.zeros(1, device=device),
                               sorted_boundaries,
                               torch.ones(1, device=device) * 2])
        
        sharpness = torch.abs(transition_sharpness) + 0.5
        gains = torch.sigmoid(band_gains)
        
        response = torch.zeros_like(norm_eigenvals)
        
        for i in range(self.n_bands):
            left_boundary = boundaries[i]
            right_boundary = boundaries[i + 1]
            
            if i == 0:
                weight = torch.sigmoid(sharpness * (right_boundary - norm_eigenvals))
            elif i == self.n_bands - 1:
                weight = torch.sigmoid(sharpness * (norm_eigenvals - left_boundary))
            else:
                left_weight = torch.sigmoid(sharpness * (norm_eigenvals - left_boundary))
                right_weight = torch.sigmoid(sharpness * (right_boundary - norm_eigenvals))
                weight = left_weight * right_weight
            
            response += gains[i] * weight
        
        global_scale_sigmoid = torch.sigmoid(global_scale) + 0.1
        response *= global_scale_sigmoid
        
        return torch.clamp(response, min=1e-6, max=1.2)

# =============================================================================
# FILTER DESIGN 9: HARMONIC SPECTRAL FILTER
# =============================================================================
class HarmonicSpectralFilter(nn.Module):
    """Filter based on harmonic series with learnable fundamental frequency"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth', n_harmonics=3):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        self.n_harmonics = n_harmonics
        
        self.fundamental_freq = nn.Parameter(torch.tensor(0.5))
        self.harmonic_weights = nn.Parameter(torch.ones(n_harmonics))
        self.harmonic_widths = nn.Parameter(torch.ones(n_harmonics) * 0.1)
        
        self.base_level = nn.Parameter(torch.tensor(0.1))
        self.decay_rate = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval * 2
        
        fundamental_freq = self.fundamental_freq.to(device)
        harmonic_weights = self.harmonic_weights.to(device)
        harmonic_widths = self.harmonic_widths.to(device)
        base_level = self.base_level.to(device)
        decay_rate = self.decay_rate.to(device)
        
        base = torch.sigmoid(base_level) * 0.3
        response = base * torch.ones_like(norm_eigenvals)
        
        fund_freq = torch.sigmoid(fundamental_freq) * 1.5 + 0.1
        
        for i in range(self.n_harmonics):
            harmonic_freq = fund_freq * (i + 1)
            
            if harmonic_freq < 2.0:
                weight = torch.sigmoid(harmonic_weights[i])
                width = torch.sigmoid(harmonic_widths[i]) * 0.3 + 0.05
                
                decay = torch.exp(-decay_rate * i * 0.5)
                
                harmonic_response = weight * decay * torch.exp(-((norm_eigenvals - harmonic_freq) / width) ** 2)
                response += harmonic_response
        
        return torch.clamp(response, min=1e-6, max=1.0)

# =============================================================================
# POLYNOMIAL FILTER CLASSES (COMPLETE SET)
# =============================================================================

class UniversalPolynomialFilter(nn.Module):
    """Universal Polynomial Filter supporting multiple polynomial bases"""
    
    def __init__(self, 
                 filter_order=6, 
                 polynomial_type='chebyshev',
                 init_filter_name='smooth',
                 learnable_coeffs=True,
                 polynomial_params=None):
        super().__init__()
        
        self.filter_order = filter_order
        self.polynomial_type = polynomial_type.lower()
        self.init_filter_name = init_filter_name
        self.learnable_coeffs = learnable_coeffs
        self.polynomial_params = polynomial_params or {}
        
        self.valid_types = {
            'chebyshev': ChebyshevBasis,
            'jacobi': JacobiBasis,
            'legendre': LegendreBasis,
            'laguerre': LaguerreBasis,
            'hermite': HermiteBasis,
            'bernstein': BernsteinBasis
        }
        
        if self.polynomial_type not in self.valid_types:
            raise ValueError(f"Polynomial type '{polynomial_type}' not supported. "
                           f"Available: {list(self.valid_types.keys())}")
        
        self.basis_class = self.valid_types[self.polynomial_type]
        
        self._initialize_coefficients()
        
        self.domain_scale = nn.Parameter(torch.tensor(1.0))
        self.domain_shift = nn.Parameter(torch.tensor(0.0))
    
    def _initialize_coefficients(self):
        try:
            initial_coeffs = get_filter_coefficients(
                self.init_filter_name, 
                order=self.filter_order, 
                as_tensor=True
            )
        except:
            initial_coeffs = torch.tensor([1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015])
            if len(initial_coeffs) < self.filter_order + 1:
                padding = torch.zeros(self.filter_order + 1 - len(initial_coeffs))
                initial_coeffs = torch.cat([initial_coeffs, padding])
            elif len(initial_coeffs) > self.filter_order + 1:
                initial_coeffs = initial_coeffs[:self.filter_order + 1]
        
        if len(initial_coeffs) < self.filter_order + 1:
            padding = torch.zeros(self.filter_order + 1 - len(initial_coeffs))
            initial_coeffs = torch.cat([initial_coeffs, padding])
        elif len(initial_coeffs) > self.filter_order + 1:
            initial_coeffs = initial_coeffs[:self.filter_order + 1]
        
        if self.learnable_coeffs:
            self.coefficients = nn.Parameter(initial_coeffs.clone())
        else:
            self.register_buffer('coefficients', initial_coeffs.clone())
    
    def _transform_domain(self, eigenvalues):
        scale = torch.abs(self.domain_scale) + 0.1
        shift = torch.tanh(self.domain_shift)
        
        if self.polynomial_type in ['chebyshev', 'jacobi', 'legendre']:
            max_eigenval = torch.max(eigenvalues) + 1e-8
            x = 2 * (eigenvalues / max_eigenval) - 1
            x = scale * x + shift
            x = torch.clamp(x, -0.99, 0.99)
            
        elif self.polynomial_type == 'laguerre':
            max_eigenval = torch.max(eigenvalues) + 1e-8
            x = eigenvalues / max_eigenval * scale + shift
            x = torch.clamp(x, 0.01, 10.0)
            
        elif self.polynomial_type == 'hermite':
            max_eigenval = torch.max(eigenvalues) + 1e-8
            x = (eigenvalues / max_eigenval - 0.5) * 2 * scale + shift
            x = torch.clamp(x, -3.0, 3.0)
            
        elif self.polynomial_type == 'bernstein':
            max_eigenval = torch.max(eigenvalues) + 1e-8
            x = eigenvalues / max_eigenval * scale + shift
            x = torch.clamp(x, 0.01, 0.99)
            
        else:
            max_eigenval = torch.max(eigenvalues) + 1e-8
            x = eigenvalues / max_eigenval
        
        return x
    
    def forward(self, eigenvalues):
        x = self._transform_domain(eigenvalues)
        
        polynomials = self.basis_class.compute_polynomials(
            x, self.filter_order, **self.polynomial_params
        )
        
        coeffs = self.coefficients.to(eigenvalues.device)
        
        filter_response = torch.sum(coeffs.unsqueeze(0) * polynomials, dim=-1)
        
        if self.polynomial_type in ['laguerre', 'hermite']:
            filter_response = torch.exp(-torch.abs(filter_response).clamp(max=10.0)) + 1e-6
        else:
            filter_response = torch.sigmoid(filter_response) + 1e-6
        
        return filter_response

class ChebyshevSpectralFilter(UniversalPolynomialFilter):
    """Chebyshev polynomial spectral filter"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__(
            filter_order=filter_order,
            polynomial_type='chebyshev',
            init_filter_name=init_filter_name,
            learnable_coeffs=True
        )

class JacobiSpectralFilter(UniversalPolynomialFilter):
    """Jacobi polynomial spectral filter with learnable alpha, beta parameters"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth', alpha=0.0, beta=0.0):
        super().__init__(
            filter_order=filter_order,
            polynomial_type='jacobi',
            init_filter_name=init_filter_name,
            learnable_coeffs=True,
            polynomial_params={'alpha': alpha, 'beta': beta}
        )
        
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
    
    def forward(self, eigenvalues):
        self.polynomial_params['alpha'] = self.alpha.item()
        self.polynomial_params['beta'] = self.beta.item()
        return super().forward(eigenvalues)

class LegendreSpectralFilter(UniversalPolynomialFilter):
    """Legendre polynomial spectral filter"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__(
            filter_order=filter_order,
            polynomial_type='legendre',
            init_filter_name=init_filter_name,
            learnable_coeffs=True
        )

class LaguerreSpectralFilter(UniversalPolynomialFilter):
    """Laguerre polynomial spectral filter"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth', alpha=0.0):
        super().__init__(
            filter_order=filter_order,
            polynomial_type='laguerre',
            init_filter_name=init_filter_name,
            learnable_coeffs=True,
            polynomial_params={'alpha': alpha}
        )

class HermiteSpectralFilter(UniversalPolynomialFilter):
    """Hermite polynomial spectral filter"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__(
            filter_order=filter_order,
            polynomial_type='hermite',
            init_filter_name=init_filter_name,
            learnable_coeffs=True
        )

class BernsteinSpectralFilter(UniversalPolynomialFilter):
    """Bernstein polynomial spectral filter"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__(
            filter_order=filter_order,
            polynomial_type='bernstein',
            init_filter_name=init_filter_name,
            learnable_coeffs=True
        )

# =============================================================================
# FILTER DESIGN 10: ENSEMBLE SPECTRAL FILTER (ENHANCED)
# =============================================================================
class EnsembleSpectralFilter(nn.Module):
    """Enhanced ensemble with ALL filter types"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        
        # Include ALL filter types
        self.universal_filter = UniversalSpectralFilter(filter_order, init_filter_name)
        self.spectral_basis_filter = SpectralBasisFilter(filter_order, init_filter_name)
        self.enhanced_filter = EnhancedSpectralBasisFilter(filter_order, init_filter_name)
        self.golden_filter = AdaptiveGoldenFilter(filter_order, init_filter_name)
        self.multiscale_filter = MultiScaleSpectralFilter(filter_order, init_filter_name)
        self.bandstop_filter = BandStopSpectralFilter(filter_order, init_filter_name)
        self.adaptive_bandstop_filter = AdaptiveBandStopFilter(filter_order, init_filter_name)
        self.parametric_filter = ParametricMultiBandFilter(filter_order, init_filter_name)
        self.harmonic_filter = HarmonicSpectralFilter(filter_order, init_filter_name)
        self.chebyshev_filter = ChebyshevSpectralFilter(filter_order, init_filter_name)
        
        self.ensemble_weights = nn.Parameter(torch.ones(10))
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        
        responses = [
            self.universal_filter(eigenvalues),
            self.spectral_basis_filter(eigenvalues),
            self.enhanced_filter(eigenvalues),
            self.golden_filter(eigenvalues),
            self.multiscale_filter(eigenvalues),
            self.bandstop_filter(eigenvalues),
            self.adaptive_bandstop_filter(eigenvalues),
            self.parametric_filter(eigenvalues),
            self.harmonic_filter(eigenvalues),
            self.chebyshev_filter(eigenvalues)
        ]
        
        ensemble_weights = self.ensemble_weights.to(device)
        temperature = self.temperature.to(device)
        temp = torch.abs(temperature) + 0.1
        weights = torch.softmax(ensemble_weights / temp, dim=0)
        
        final_response = sum(w * resp for w, resp in zip(weights, responses))
        return final_response

# =============================================================================
# FILTER FACTORY (COMPLETE)
# =============================================================================

def create_filter(filter_design, filter_order=6, init_filter_name='smooth', **kwargs):
    """Factory function to create ANY filter"""
    
    filter_map = {
        # Core filters
        'original': UniversalSpectralFilter,
        'spectral_basis': SpectralBasisFilter,
        'enhanced_basis': EnhancedSpectralBasisFilter,
        
        # Polynomial filters
        'chebyshev': ChebyshevSpectralFilter,
        'jacobi': JacobiSpectralFilter,
        'legendre': LegendreSpectralFilter,
        'laguerre': LaguerreSpectralFilter,
        'hermite': HermiteSpectralFilter,
        'bernstein': BernsteinSpectralFilter,
        'universal_polynomial': UniversalPolynomialFilter,
        
        # Advanced filters
        'golden': AdaptiveGoldenFilter,
        'multiscale': MultiScaleSpectralFilter,
        'bandstop': BandStopSpectralFilter,
        'adaptive_bandstop': AdaptiveBandStopFilter,
        'parametric': ParametricMultiBandFilter,
        'harmonic': HarmonicSpectralFilter,
        
        # Ensemble
        'ensemble': EnsembleSpectralFilter,
    }
    
    if filter_design not in filter_map:
        print(f"⚠️ Unknown filter design '{filter_design}', using 'enhanced_basis'")
        filter_design = 'enhanced_basis'
    
    filter_class = filter_map[filter_design]
    
    # Handle different constructor signatures
    if filter_design == 'parametric':
        return filter_class(filter_order, init_filter_name, kwargs.get('n_bands', 4))
    elif filter_design == 'multiscale':
        return filter_class(filter_order, init_filter_name, kwargs.get('n_bands', 4))
    elif filter_design == 'harmonic':
        return filter_class(filter_order, init_filter_name, kwargs.get('n_harmonics', 3))
    elif filter_design == 'adaptive_bandstop':
        return filter_class(filter_order, init_filter_name, kwargs.get('n_stop_bands', 2))
    elif filter_design == 'jacobi':
        return filter_class(filter_order, init_filter_name, kwargs.get('alpha', 0.0), kwargs.get('beta', 0.0))
    elif filter_design == 'laguerre':
        return filter_class(filter_order, init_filter_name, kwargs.get('alpha', 0.0))
    elif filter_design == 'universal_polynomial':
        return filter_class(
            filter_order, 
            kwargs.get('polynomial_type', 'chebyshev'),
            init_filter_name,
            kwargs.get('learnable_coeffs', True),
            kwargs.get('polynomial_params', None)
        )
    else:
        return filter_class(filter_order, init_filter_name)

# Backward compatibility
SpectralBasisFilter = SpectralBasisFilter