#!/usr/bin/env python3
"""
Simple test for temperature scaling effect
"""
import numpy as np
import scipy.sparse as sp

def test_temperature_effect():
    """Demonstrate temperature scaling effect on similarities"""
    print("=" * 60)
    print("Temperature Scaling Effect on User Similarities")
    print("=" * 60)
    
    # Create a small example similarity matrix
    # Values range from 0.1 to 0.9 to simulate different similarity strengths
    similarities = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    temperatures = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    print("Original similarities:", similarities)
    print()
    
    for temp in temperatures:
        # Apply temperature scaling
        scaled = np.exp(similarities / temp)
        
        print(f"Temperature = {temp}:")
        print(f"  Scaled values: {scaled}")
        print(f"  Min: {np.min(scaled):.3f}, Max: {np.max(scaled):.3f}")
        print(f"  Ratio (max/min): {np.max(scaled)/np.min(scaled):.1f}")
        print(f"  Normalized: {scaled / np.sum(scaled)}")
        print()
    
    print("\nObservations:")
    print("- Lower temperature (0.01) creates extreme amplification")
    print("- Temperature 0.1 provides good discrimination")
    print("- Higher temperature (1.0) preserves relative differences")
    print("\nFor sparse user-item data, temperature ~0.1 should help")
    print("distinguish truly similar users from noise.")

if __name__ == "__main__":
    test_temperature_effect()