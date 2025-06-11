'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
DySimGCF-Style Implementation with True Similarity-Based Graph Construction

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import os
import sys
import argparse
import time
import warnings
import subprocess
import json
from pathlib import Path
from collections import defaultdict
import statistics
import numpy as np

warnings.filterwarnings("ignore")

def run_single_experiment(dataset, filter_design, init_filter, quick_mode=False, verbose=False):
    """Run a single experiment with given dataset, filter design and initialization"""
    
    # Dataset-specific parameters
    dataset_params = {
        'ml-100k': {
            'n_eigen': '50' if quick_mode else '100',
            'epochs': '30' if quick_mode else '50',
            'train_u_batch_size': '500',
            'eval_u_batch_size': '200'
        },
        'ml-1m': {
            'n_eigen': '100' if quick_mode else '200',
            'epochs': '25' if quick_mode else '40',
            'train_u_batch_size': '1000',
            'eval_u_batch_size': '400'
        },
        'lastfm': {
            'n_eigen': '80' if quick_mode else '150',
            'epochs': '25' if quick_mode else '40',
            'train_u_batch_size': '800',
            'eval_u_batch_size': '300'
        }
    }
    
    params = dataset_params.get(dataset, dataset_params['ml-100k'])
    
    cmd = [
        sys.executable, "main.py",
        "--dataset", dataset,
        "--filter_design", filter_design,  # This is the ARCHITECTURE
        "--init_filter", init_filter,      # This is the INITIALIZATION PATTERN
        "--filter", "ui",
        "--filter_order", "6",
        "--lr", "0.001",
        "--decay", "0.01",
        "--n_eigen", params['n_eigen'],
        "--epochs", params['epochs'],
        "--patience", "8",
        "--train_u_batch_size", params['train_u_batch_size'],
        "--eval_u_batch_size", params['eval_u_batch_size'],
        "--seed", "2025",
        "--verbose", "0"
    ]
    
    try:
        print(f"  Testing {filter_design:15} + {init_filter:15}...", end=" ", flush=True)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        
        if result.returncode != 0:
            print("❌ FAILED")
            if verbose:
                print(f"    Error: {result.stderr[:200]}")
            return None
        
        # Parse NDCG from output
        output_lines = result.stdout.split('\n')
        ndcg_value = None
        
        for line in output_lines:
            if "Final Test Results:" in line and "NDCG@20=" in line:
                try:
                    ndcg_part = line.split("NDCG@20=")[1]
                    ndcg_value = float(ndcg_part.split(",")[0])
                    break
                except (IndexError, ValueError):
                    continue
        
        if ndcg_value is not None:
            print(f"✅ NDCG@20: {ndcg_value:.6f}")
            return ndcg_value
        else:
            print("❌ NO RESULT")
            if verbose:
                print(f"    Output preview: {result.stdout[-500:]}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT")
        return None
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None

def get_filter_combinations():
    """Get CORRECT filter design and initialization combinations"""
    
    # CORRECTED: These are the actual ARCHITECTURES
    filter_designs = [
        'original',           # ~14 params
        'basis',             # ~32 params  
        'enhanced_basis',    # ~52 params
        'adaptive_golden',   # ~28 params
        'adaptive',          # ~15 params
        'neural',            # ~561 params
        'deep',              # ~1000+ params
        'multiscale',        # ~500+ params
        'ensemble'           # ~2000+ params
    ]
    
    # CORRECTED: These are the actual INITIALIZATION PATTERNS from filters.py
    init_filters = [
        # Core smoothing filters
        'smooth', 'butterworth', 'gaussian', 'bessel', 'conservative',
        
        # Golden ratio variants (high-performance)
        'golden_034', 'golden_036', 'golden_348', 'golden_352',
        'soft_golden_ratio', 'golden_ratio_balanced', 
        'golden_optimized_1', 'golden_optimized_2', 'golden_optimized_3',
        
        # Oscillatory patterns
        'oscillatory_soft', 'oscillatory_soft_v2', 'oscillatory_soft_v3',
        
        # Fine-tuned coefficients
        'soft_tuned_351', 'soft_tuned_352', 'soft_tuned_353',
        
        # Mathematical patterns
        'fibonacci_soft', 'euler_soft', 'natural_harmony',
        
        # Multi-band and adaptive patterns
        'multi_band', 'multi_band_balanced', 'wiener_like', 'adaptive_smooth',
        
        # Baselines
        'identity', 'exponential_decay'
    ]
    
    return filter_designs, init_filters

def get_filter_design_categories():
    """Categorize filter designs by parameter count and complexity"""
    return {
        'Low Complexity (<50 params)': ['original', 'adaptive', 'adaptive_golden', 'basis'],
        'Medium Complexity (50-100 params)': ['enhanced_basis'],
        'High Complexity (500+ params)': ['neural', 'multiscale'],
        'Very High Complexity (1000+ params)': ['deep', 'ensemble']
    }

def get_init_filter_categories():
    """Categorize initialization filters by type"""
    return {
        'Smoothing': ['smooth', 'butterworth', 'gaussian', 'bessel', 'conservative'],
        'Golden Ratio': ['golden_034', 'golden_036', 'golden_348', 'golden_352',
                        'soft_golden_ratio', 'golden_ratio_balanced',
                        'golden_optimized_1', 'golden_optimized_2', 'golden_optimized_3'],
        'Oscillatory': ['oscillatory_soft', 'oscillatory_soft_v2', 'oscillatory_soft_v3'],
        'Fine-tuned': ['soft_tuned_351', 'soft_tuned_352', 'soft_tuned_353'],
        'Mathematical': ['fibonacci_soft', 'euler_soft', 'natural_harmony'],
        'Adaptive': ['multi_band', 'multi_band_balanced', 'wiener_like', 'adaptive_smooth'],
        'Baseline': ['identity', 'exponential_decay']
    }

def main():
    parser = argparse.ArgumentParser(description="CORRECTED Filter Performance Tester")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'ml-1m', 'lastfm'],
                       help='Dataset to test on')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer epochs')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output for failed runs')
    parser.add_argument('--test_mode', type=str, default='efficiency',
                       choices=['efficiency', 'architecture', 'initialization', 'full'],
                       help='Test mode: efficiency (best combos), architecture (all designs), initialization (all inits), full (everything)')
    
    args = parser.parse_args()
    
    print("🚀 CORRECTED Universal Filter Performance Tester")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset.upper()}")
    print(f"🎯 Test Mode: {args.test_mode.upper()}")
    
    filter_designs, init_filters = get_filter_combinations()
    
    # Select combinations based on test mode
    if args.test_mode == 'efficiency':
        # Test best performing combinations only
        test_combinations = [
            ('basis', 'golden_036'),
            ('enhanced_basis', 'soft_golden_ratio'), 
            ('adaptive_golden', 'golden_optimized_1'),
            ('neural', 'smooth'),
            ('deep', 'butterworth'),
            ('ensemble', 'golden_ratio_balanced')
        ]
        print(f"🎯 Testing {len(test_combinations)} high-efficiency combinations")
        
    elif args.test_mode == 'architecture':
        # Test all architectures with best initialization
        best_init = 'golden_036'
        test_combinations = [(design, best_init) for design in filter_designs]
        print(f"🔧 Testing all {len(filter_designs)} architectures with {best_init} initialization")
        
    elif args.test_mode == 'initialization':
        # Test all initializations with best architecture
        best_design = 'enhanced_basis'
        test_combinations = [(best_design, init) for init in init_filters]
        print(f"🎛️ Testing all {len(init_filters)} initializations with {best_design} architecture")
        
    else:  # full
        # Test everything (WARNING: This is HUGE!)
        test_combinations = [(design, init) for design in filter_designs for init in init_filters]
        total = len(test_combinations)
        print(f"🌟 FULL TEST: {total} combinations ({len(filter_designs)} designs × {len(init_filters)} inits)")
        
        if total > 50:
            response = input(f"⚠️ This will run {total} experiments. Continue? (y/N): ")
            if response.lower() != 'y':
                print("❌ Test cancelled.")
                return 1
    
    # Run experiments
    results = []
    start_time = time.time()
    
    print(f"\n🏃 Running {len(test_combinations)} experiments...")
    print("-" * 60)
    
    for i, (filter_design, init_filter) in enumerate(test_combinations, 1):
        print(f"\n[{i}/{len(test_combinations)}] {filter_design.upper()} + {init_filter}")
        
        ndcg = run_single_experiment(
            args.dataset, filter_design, init_filter,
            quick_mode=args.quick, verbose=args.verbose
        )
        
        if ndcg is not None:
            results.append({
                'filter_design': filter_design,
                'init_filter': init_filter,
                'ndcg': ndcg,
                'combination': f"{filter_design}+{init_filter}"
            })
    
    # Analyze results
    if not results:
        print("❌ No successful experiments!")
        return 1
    
    results.sort(key=lambda x: x['ndcg'], reverse=True)
    
    print("\n" + "=" * 80)
    print(f"🏆 RESULTS - {args.dataset.upper()}")
    print("=" * 80)
    
    print(f"\n🥇 TOP 10 COMBINATIONS:")
    print(f"{'Rank':<4} {'Design':<15} {'Initialization':<20} {'NDCG@20':<10}")
    print("-" * 55)
    
    for i, result in enumerate(results[:10], 1):
        print(f"{i:<4} {result['filter_design']:<15} {result['init_filter']:<20} {result['ndcg']:<10.6f}")
    
    # Architecture analysis
    print(f"\n📊 ARCHITECTURE ANALYSIS:")
    design_performance = defaultdict(list)
    for result in results:
        design_performance[result['filter_design']].append(result['ndcg'])
    
    for design in filter_designs:
        if design in design_performance:
            scores = design_performance[design]
            print(f"  {design:<15}: Best={max(scores):.6f}, Mean={np.mean(scores):.6f}, Count={len(scores)}")
    
    # Initialization analysis
    print(f"\n🎛️ INITIALIZATION ANALYSIS:")
    init_performance = defaultdict(list)
    for result in results:
        init_performance[result['init_filter']].append(result['ndcg'])
    
    init_categories = get_init_filter_categories()
    for category, filters in init_categories.items():
        category_scores = []
        for init_filter in filters:
            if init_filter in init_performance:
                category_scores.extend(init_performance[init_filter])
        
        if category_scores:
            print(f"  {category:<15}: Best={max(category_scores):.6f}, Mean={np.mean(category_scores):.6f}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"✅ Completed {len(results)} successful experiments")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

# ============================================================================
# CORRECTED Filter Performance Tester - Usage Examples
# ============================================================================

# # Efficiency Testing (Recommended Start)
# python filter_test_corrected.py --dataset ml-100k --test_mode efficiency
# python filter_test_corrected.py --dataset lastfm --test_mode efficiency --quick

# # Architecture Comparison (Test all filter designs with same initialization)
# python filter_test_corrected.py --dataset ml-100k --test_mode architecture
# python filter_test_corrected.py --dataset ml-1m --test_mode architecture --quick

# # Initialization Robustness (Test all init patterns with same architecture)
# python filter_test_corrected.py --dataset ml-100k --test_mode initialization
# python filter_test_corrected.py --dataset lastfm --test_mode initialization --quick

# # Full Comparison (WARNING: Very long!)
# python filter_test_corrected.py --dataset ml-100k --test_mode full --quick
# python filter_test_corrected.py --dataset ml-1m --test_mode full  # Only for serious analysis

# # Debugging Failed Runs
# python filter_test_corrected.py --dataset ml-100k --test_mode efficiency --verbose

# ============================================================================
# What Each Test Mode Does:
# ============================================================================

# EFFICIENCY MODE (--test_mode efficiency):
# Tests carefully selected high-performance combinations:
# - basis + golden_036           (balanced performance)
# - enhanced_basis + soft_golden_ratio  (maximum traditional performance)  
# - adaptive_golden + golden_optimized_1 (golden ratio optimization)
# - neural + smooth              (neural approach baseline)
# - deep + butterworth           (high-capacity neural)
# - ensemble + golden_ratio_balanced    (maximum capacity)

# ARCHITECTURE MODE (--test_mode architecture):
# Tests ALL filter designs with the same initialization (golden_036):
# - original + golden_036
# - basis + golden_036  
# - enhanced_basis + golden_036
# - adaptive_golden + golden_036
# - adaptive + golden_036
# - neural + golden_036
# - deep + golden_036
# - multiscale + golden_036
# - ensemble + golden_036
# → Shows which ARCHITECTURE performs best

# INITIALIZATION MODE (--test_mode initialization): 
# Tests ALL initialization patterns with the same architecture (enhanced_basis):
# - enhanced_basis + smooth
# - enhanced_basis + golden_036
# - enhanced_basis + fibonacci_soft
# - enhanced_basis + oscillatory_soft
# - ... (all 25+ initialization patterns)
# → Shows which INITIALIZATION works best

# FULL MODE (--test_mode full):
# Tests EVERY combination (9 designs × 25+ patterns = 225+ experiments!)
# - Only use for comprehensive research
# - Recommended with --quick flag
# - Takes hours to complete

# ============================================================================
# Expected Results Interpretation:
# ============================================================================

# TOP PERFORMERS (typically):
# 1. enhanced_basis + soft_golden_ratio  (~0.392 NDCG on ml-100k)
# 2. adaptive_golden + golden_optimized_1 (~0.391 NDCG)  
# 3. basis + golden_036                   (~0.390 NDCG)

# ARCHITECTURE RANKING (by complexity vs performance):
# 1. enhanced_basis  (~52 params)  - Best performance/complexity ratio
# 2. adaptive_golden (~28 params)  - Most efficient  
# 3. basis          (~32 params)  - Good baseline
# 4. neural         (~561 params) - Diminishing returns start here
# 5. deep           (~1000+ params) - Often overfits on small datasets
# 6. ensemble       (~2000+ params) - Only good on very large datasets

# INITIALIZATION RANKING (by category):
# 1. Golden Ratio variants - Consistently best (golden_036, soft_golden_ratio)
# 2. Fine-tuned patterns  - Good performance (soft_tuned_351, soft_tuned_352)  
# 3. Smoothing filters    - Reliable baseline (smooth, butterworth, gaussian)
# 4. Mathematical patterns - Variable results (fibonacci_soft, euler_soft)
# 5. Baseline patterns    - Worst performance (identity, exponential_decay)

# ============================================================================
# Troubleshooting:
# ============================================================================

# If you get "Unknown filter pattern" errors:
# - Check that init_filter exists in filters.py filter_patterns dictionary
# - Use --verbose flag to see detailed error messages
# - Verify you're using initialization patterns, not filter designs

# If experiments timeout:
# - Use --quick flag to reduce epochs
# - Reduce batch sizes for high-capacity filters
# - Test on smaller datasets first (ml-100k before ml-1m)

# If performance is unexpectedly low:
# - Check dataset is loaded correctly
# - Verify eigenvalue decomposition succeeded  
# - Use --verbose to see training progress

# ============================================================================