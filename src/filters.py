'''
Created on June 7, 2025
Enhanced Universal Spectral Filter Patterns with Band-Stop and Advanced Filters
FIXED: Device compatibility issues resolved

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import torch
import torch.nn as nn
import numpy as np

# Essential filter patterns (cleaned up)
filter_patterns = {
    # Core smoothing filters
    'smooth': [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015],
    'butterworth': [1.0, -0.6, 0.2, -0.05, 0.01, -0.002, 0.0003],
    'gaussian': [1.0, -0.7, 0.15, -0.03, 0.005, -0.0007, 0.00008],
    
    # Golden ratio variants (proven effective)
    'golden_036': [1.0, -0.36, 0.1296, -0.220, 0.1564, -0.088, 0.0548],
    
    # Band-stop patterns (pass low and high, reject middle)
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
# FILTER DESIGN 1: ORIGINAL UNIVERSAL FILTER
# =============================================================================
class UniversalSpectralFilter(nn.Module):
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        # Initialize coefficients
        lowpass = get_filter_coefficients(init_filter_name, as_tensor=True)
        coeffs_data = torch.zeros(filter_order + 1)
        for i, val in enumerate(lowpass[:filter_order + 1]):
            coeffs_data[i] = val

        self.register_buffer('init_coeffs', coeffs_data.clone())
        self.coeffs = nn.Parameter(coeffs_data.clone())
    
    def forward(self, eigenvalues):
        # Ensure coeffs is on the same device as eigenvalues
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
        
        # Register filter coefficients as buffers for proper device handling
        for i, name in enumerate(filter_names):
            coeffs = get_filter_coefficients(name, order=filter_order, as_tensor=True)
            if len(coeffs) < filter_order + 1:
                padded_coeffs = torch.zeros(filter_order + 1)
                padded_coeffs[:len(coeffs)] = coeffs
                coeffs = padded_coeffs
            elif len(coeffs) > filter_order + 1:
                coeffs = coeffs[:filter_order + 1]
            
            self.register_buffer(f'filter_{i}', coeffs)
        
        # Store filter names for reference
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
        
        # Get filter bank on the correct device
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
    
    def get_mixing_analysis(self):
        weights = torch.softmax(self.mixing_weights, dim=0).detach().cpu().numpy()
        analysis = {}
        for i, name in enumerate(self.filter_names):
            analysis[name] = weights[i]
        return analysis

# =============================================================================
# FILTER DESIGN 3: ENHANCED SPECTRAL BASIS FILTER
# =============================================================================
class EnhancedSpectralBasisFilter(nn.Module):
    """Enhanced basis filter for maximum performance"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        # Extended filter bank with more golden variants
        filter_names = ['golden_036', 'smooth', 'butterworth', 'gaussian']
        
        # Register filter coefficients as buffers for proper device handling
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
        
        # Store valid filter names
        self.filter_names = filter_names[:4]  # We know we have 4 filters
        
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
        
        # Get filter bank on the correct device
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
    
    def get_mixing_analysis(self):
        weights = torch.softmax(self.mixing_weights, dim=0).detach().cpu().numpy()
        analysis = {}
        for i, name in enumerate(self.filter_names):
            analysis[name] = weights[i]
        
        sorted_analysis = dict(sorted(analysis.items(), key=lambda x: x[1], reverse=True))
        return sorted_analysis

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
        
        # Learnable band-stop parameters
        self.low_cutoff = nn.Parameter(torch.tensor(0.2))
        self.high_cutoff = nn.Parameter(torch.tensor(1.0))
        self.low_gain = nn.Parameter(torch.tensor(1.0))
        self.high_gain = nn.Parameter(torch.tensor(0.8))
        self.stop_depth = nn.Parameter(torch.tensor(0.05))
        self.transition_sharpness = nn.Parameter(torch.tensor(5.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        
        # Move all parameters to the correct device
        low_cutoff = self.low_cutoff.to(device)
        high_cutoff = self.high_cutoff.to(device)
        low_gain = self.low_gain.to(device)
        high_gain = self.high_gain.to(device)
        stop_depth = self.stop_depth.to(device)
        transition_sharpness = self.transition_sharpness.to(device)
        
        # Ensure proper parameter ranges
        low_cut = torch.sigmoid(low_cutoff) * 0.8
        high_cut = low_cut + torch.sigmoid(high_cutoff) * (2.0 - low_cut)
        low_g = torch.sigmoid(low_gain)
        high_g = torch.sigmoid(high_gain)
        stop_d = torch.sigmoid(stop_depth) * 0.2
        sharpness = torch.abs(transition_sharpness) + 1.0
        
        # Normalize eigenvalues to [0, 1] for easier processing
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval
        norm_low_cut = low_cut / 2.0
        norm_high_cut = high_cut / 2.0
        
        # Create band-stop response
        response = torch.ones_like(norm_eigenvals)
        
        # Low-pass component (pass low frequencies)
        low_mask = norm_eigenvals < norm_low_cut
        response = torch.where(low_mask, low_g * response, response)
        
        # High-pass component (pass high frequencies)  
        high_mask = norm_eigenvals > norm_high_cut
        response = torch.where(high_mask, high_g * response, response)
        
        # Stop band (reject middle frequencies)
        stop_mask = (norm_eigenvals >= norm_low_cut) & (norm_eigenvals <= norm_high_cut)
        if stop_mask.any():
            # Smooth transition in stop band
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
        
        # Multiple stop bands with learnable parameters
        self.band_centers = nn.Parameter(torch.linspace(0.3, 1.2, n_stop_bands))
        self.band_widths = nn.Parameter(torch.ones(n_stop_bands) * 0.2)
        self.band_depths = nn.Parameter(torch.ones(n_stop_bands) * 0.1)
        
        # Global parameters
        self.base_gain = nn.Parameter(torch.tensor(1.0))
        self.high_freq_boost = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval * 2  # Scale to [0, 2]
        
        # Move parameters to device
        band_centers = self.band_centers.to(device)
        band_widths = self.band_widths.to(device)
        band_depths = self.band_depths.to(device)
        base_gain = self.base_gain.to(device)
        high_freq_boost = self.high_freq_boost.to(device)
        
        # Start with base response
        base_g = torch.sigmoid(base_gain)
        response = base_g * torch.ones_like(norm_eigenvals)
        
        # Apply each stop band
        for i in range(self.n_stop_bands):
            center = torch.clamp(band_centers[i], 0.1, 1.9)
            width = torch.sigmoid(band_widths[i]) * 0.5 + 0.05
            depth = torch.sigmoid(band_depths[i]) * 0.8
            
            # Gaussian-shaped stop band
            band_attenuation = 1 - depth * torch.exp(-((norm_eigenvals - center) / width) ** 2)
            response *= band_attenuation
        
        # Optional high frequency boost
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
        
        # Learnable band boundaries (frequencies)
        init_boundaries = torch.linspace(0.2, 1.8, n_bands - 1)
        self.band_boundaries = nn.Parameter(init_boundaries)
        
        # Learnable gains for each band
        self.band_gains = nn.Parameter(torch.ones(n_bands))
        
        # Transition parameters
        self.transition_sharpness = nn.Parameter(torch.tensor(3.0))
        self.global_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval * 2  # Scale to [0, 2]
        
        # Move parameters to device
        band_boundaries = self.band_boundaries.to(device)
        band_gains = self.band_gains.to(device)
        transition_sharpness = self.transition_sharpness.to(device)
        global_scale = self.global_scale.to(device)
        
        # Sort boundaries to ensure proper ordering
        sorted_boundaries = torch.sort(band_boundaries)[0]
        boundaries = torch.cat([torch.zeros(1, device=device),
                               sorted_boundaries,
                               torch.ones(1, device=device) * 2])
        
        # Smooth band transitions
        sharpness = torch.abs(transition_sharpness) + 0.5
        gains = torch.sigmoid(band_gains)
        
        response = torch.zeros_like(norm_eigenvals)
        
        # Calculate response for each band
        for i in range(self.n_bands):
            left_boundary = boundaries[i]
            right_boundary = boundaries[i + 1]
            
            # Smooth transitions at boundaries
            if i == 0:
                # First band: ramp from left
                weight = torch.sigmoid(sharpness * (right_boundary - norm_eigenvals))
            elif i == self.n_bands - 1:
                # Last band: ramp to right
                weight = torch.sigmoid(sharpness * (norm_eigenvals - left_boundary))
            else:
                # Middle bands: bell-shaped
                left_weight = torch.sigmoid(sharpness * (norm_eigenvals - left_boundary))
                right_weight = torch.sigmoid(sharpness * (right_boundary - norm_eigenvals))
                weight = left_weight * right_weight
            
            response += gains[i] * weight
        
        # Apply global scaling
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
        
        # Learnable fundamental frequency and harmonics
        self.fundamental_freq = nn.Parameter(torch.tensor(0.5))
        self.harmonic_weights = nn.Parameter(torch.ones(n_harmonics))
        self.harmonic_widths = nn.Parameter(torch.ones(n_harmonics) * 0.1)
        
        # Global parameters
        self.base_level = nn.Parameter(torch.tensor(0.1))
        self.decay_rate = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval * 2  # Scale to [0, 2]
        
        # Move parameters to device
        fundamental_freq = self.fundamental_freq.to(device)
        harmonic_weights = self.harmonic_weights.to(device)
        harmonic_widths = self.harmonic_widths.to(device)
        base_level = self.base_level.to(device)
        decay_rate = self.decay_rate.to(device)
        
        # Base level
        base = torch.sigmoid(base_level) * 0.3
        response = base * torch.ones_like(norm_eigenvals)
        
        # Fundamental frequency
        fund_freq = torch.sigmoid(fundamental_freq) * 1.5 + 0.1  # 0.1 to 1.6
        
        # Add harmonic peaks
        for i in range(self.n_harmonics):
            harmonic_freq = fund_freq * (i + 1)  # Fundamental, 2nd harmonic, 3rd harmonic, etc.
            
            if harmonic_freq < 2.0:  # Only if within eigenvalue range
                weight = torch.sigmoid(harmonic_weights[i])
                width = torch.sigmoid(harmonic_widths[i]) * 0.3 + 0.05  # 0.05 to 0.35
                
                # Decay for higher harmonics
                decay = torch.exp(-decay_rate * i * 0.5)
                
                # Gaussian peak at harmonic frequency
                harmonic_response = weight * decay * torch.exp(-((norm_eigenvals - harmonic_freq) / width) ** 2)
                response += harmonic_response
        
        return torch.clamp(response, min=1e-6, max=1.0)

# =============================================================================
# FILTER DESIGN 10: ENSEMBLE SPECTRAL FILTER (Enhanced)
# =============================================================================
class EnsembleSpectralFilter(nn.Module):
    """Enhanced ensemble with band-stop and parametric filters"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        self.init_filter_name = init_filter_name
        
        # Include new filter types in ensemble
        self.classical_filter = UniversalSpectralFilter(filter_order, init_filter_name)
        self.basis_filter = SpectralBasisFilter(filter_order, init_filter_name)
        self.golden_filter = AdaptiveGoldenFilter(filter_order, init_filter_name)
        self.bandstop_filter = BandStopSpectralFilter(filter_order, init_filter_name)
        self.parametric_filter = ParametricMultiBandFilter(filter_order, init_filter_name)
        
        self.ensemble_logits = nn.Parameter(torch.ones(5))
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, eigenvalues):
        device = eigenvalues.device
        
        classical_response = self.classical_filter(eigenvalues)
        basis_response = self.basis_filter(eigenvalues)
        golden_response = self.golden_filter(eigenvalues)
        bandstop_response = self.bandstop_filter(eigenvalues)
        parametric_response = self.parametric_filter(eigenvalues)
        
        # Temperature-scaled softmax for ensemble weights
        ensemble_logits = self.ensemble_logits.to(device)
        temperature = self.temperature.to(device)
        temp = torch.abs(temperature) + 0.1
        ensemble_weights = torch.softmax(ensemble_logits / temp, dim=0)
        
        final_response = (ensemble_weights[0] * classical_response +
                         ensemble_weights[1] * basis_response +
                         ensemble_weights[2] * golden_response +
                         ensemble_weights[3] * bandstop_response +
                         ensemble_weights[4] * parametric_response)
        
        return final_response
    
    def get_ensemble_analysis(self):
        with torch.no_grad():
            temp = torch.abs(self.temperature) + 0.1
            weights = torch.softmax(self.ensemble_logits / temp, dim=0)
            return {
                'classical': weights[0].item(),
                'basis': weights[1].item(), 
                'golden': weights[2].item(),
                'bandstop': weights[3].item(),
                'parametric': weights[4].item(),
                'temperature': temp.item()
            }

# =============================================================================
# EXPORT ALL FILTER CLASSES
# =============================================================================
__all__ = [
    'UniversalSpectralFilter',
    'SpectralBasisFilter', 
    'EnhancedSpectralBasisFilter',
    'AdaptiveGoldenFilter',
    'MultiScaleSpectralFilter',
    'EnsembleSpectralFilter',
    'BandStopSpectralFilter',
    'AdaptiveBandStopFilter', 
    'ParametricMultiBandFilter',
    'HarmonicSpectralFilter',
    'get_filter_coefficients',
    'filter_patterns'
]