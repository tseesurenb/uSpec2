#!/usr/bin/env python3
"""
Comprehensive UIB Filter Testing
Tests all filter combinations including the new UIB (User + Item + Bipartite) support
"""

import subprocess
import sys
import time
import re
import argparse
from pathlib import Path

def parse_ndcg(output):
    """Parse NDCG@20 from command output"""
    for line in output.split('\n'):
        if "Final Test Results:" in line and "NDCG@20=" in line:
            try:
                ndcg_part = line.split("NDCG@20=")[1]
                return float(ndcg_part.split(",")[0])
            except (IndexError, ValueError):
                continue
    return None

def extract_time(output):
    """Extract execution time from output"""
    for line in output.split('\n'):
        if "Training time:" in line:
            try:
                time_match = re.search(r'(\d+\.\d+)s', line)
                if time_match:
                    return f"{time_match.group(1)}s"
            except:
                pass
    return "?s"

def run_filter_test(model_type, dataset, filter_type, epochs=5, timeout=180):
    """Run a single filter test"""
    cmd = [
        sys.executable, "main.py",
        "--model_type", model_type,
        "--dataset", dataset,
        "--filter", filter_type,
        "--epochs", str(epochs),
        "--patience", "3",
        "--verbose", "1"
    ]
    
    # Add model-specific parameters
    if model_type == "user_specific":
        cmd.extend(["--shared_base", "--personalization_dim", "8"])
    
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0:
            ndcg = parse_ndcg(result.stdout)
            if ndcg is not None:
                execution_time = extract_time(result.stdout)
                return ndcg, execution_time, None
            else:
                return None, None, "Could not parse NDCG"
        else:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            return None, None, f"Return code {result.returncode}: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return None, None, f"Timeout ({timeout}s)"
    except Exception as e:
        return None, None, str(e)

def test_comprehensive_uib():
    """Comprehensive UIB testing"""
    print("🚀 COMPREHENSIVE UIB FILTER TESTING")
    print("=" * 80)
    
    # Test configuration
    dataset = 'ml-100k'
    epochs = 5
    
    # All filter types to test
    filters = ['u', 'i', 'b', 'ui', 'ub', 'uib']
    models = ['enhanced', 'user_specific']
    
    results = {}
    
    # Test each model with each filter
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"🔧 TESTING {model_type.upper()} MODEL")
        print(f"{'='*60}")
        
        results[model_type] = {}
        
        for filter_type in filters:
            print(f"\n🧪 Testing {model_type} model with '{filter_type}' filter...")
            
            ndcg, exec_time, error = run_filter_test(model_type, dataset, filter_type, epochs)
            
            if ndcg is not None:
                print(f"✅ SUCCESS: {model_type} + {filter_type} → NDCG@20: {ndcg:.6f} ({exec_time})")
                results[model_type][filter_type] = ndcg
            else:
                print(f"❌ FAILED: {error}")
                results[model_type][filter_type] = None
    
    # Analysis
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE UIB ANALYSIS")
    print(f"{'='*80}")
    
    for model_type in models:
        print(f"\n🔧 {model_type.upper()} MODEL RESULTS:")
        model_results = results[model_type]
        successful = {k: v for k, v in model_results.items() if v is not None}
        
        if len(successful) >= 3:
            # Individual filters
            individual_results = {k: v for k, v in successful.items() if k in ['u', 'i', 'b']}
            print(f"   Individual Filters:")
            for filter_name, score in individual_results.items():
                print(f"     {filter_name.upper():<3}: {score:.6f}")
            
            # Combination filters
            combination_results = {k: v for k, v in successful.items() if k in ['ui', 'ub', 'uib']}
            print(f"   Combination Filters:")
            for filter_name, score in combination_results.items():
                print(f"     {filter_name.upper():<3}: {score:.6f}")
            
            # Analysis
            if individual_results and combination_results:
                best_individual = max(individual_results.values())
                best_combination = max(combination_results.values())
                best_combo_name = max(combination_results.items(), key=lambda x: x[1])[0]
                
                improvement = best_combination - best_individual
                print(f"   📈 Performance Analysis:")
                print(f"     Best Individual: {best_individual:.6f}")
                print(f"     Best Combination ({best_combo_name.upper()}): {best_combination:.6f}")
                print(f"     Improvement: {improvement:+.6f}")
                
                # UIB specific analysis
                if 'uib' in combination_results:
                    uib_score = combination_results['uib']
                    if uib_score == best_combination:
                        print(f"     🏆 UIB is the BEST combination filter!")
                    else:
                        uib_vs_best = uib_score - best_combination
                        print(f"     📊 UIB vs best combination: {uib_vs_best:+.6f}")
        else:
            print(f"   ❌ Insufficient successful results: {len(successful)}/6")
    
    # Cross-model comparison
    print(f"\n🔍 CROSS-MODEL COMPARISON:")
    
    for filter_type in filters:
        enhanced_score = results.get('enhanced', {}).get(filter_type)
        user_spec_score = results.get('user_specific', {}).get(filter_type)
        
        if enhanced_score is not None and user_spec_score is not None:
            improvement = user_spec_score - enhanced_score
            print(f"   {filter_type.upper():<3}: Enhanced={enhanced_score:.6f}, User-Specific={user_spec_score:.6f} ({improvement:+.6f})")
        elif enhanced_score is not None:
            print(f"   {filter_type.upper():<3}: Enhanced={enhanced_score:.6f}, User-Specific=FAILED")
        elif user_spec_score is not None:
            print(f"   {filter_type.upper():<3}: Enhanced=FAILED, User-Specific={user_spec_score:.6f}")
        else:
            print(f"   {filter_type.upper():<3}: Both FAILED")
    
    # Final assessment
    print(f"\n🎯 FINAL UIB ASSESSMENT:")
    
    # Check if UIB works
    uib_working = False
    uib_best = False
    
    for model_type in models:
        if 'uib' in results[model_type] and results[model_type]['uib'] is not None:
            uib_working = True
            
            # Check if UIB is best for this model
            model_results = {k: v for k, v in results[model_type].items() if v is not None}
            if model_results and results[model_type]['uib'] == max(model_results.values()):
                uib_best = True
    
    if uib_working:
        print(f"   ✅ UIB Filter: WORKING")
        if uib_best:
            print(f"   🏆 UIB Performance: BEST in at least one model")
        else:
            print(f"   📊 UIB Performance: Working but not best")
    else:
        print(f"   ❌ UIB Filter: NOT WORKING")
    
    # Scaling consistency check
    b_scores = [results[model]['b'] for model in models if results[model].get('b') is not None]
    ui_scores = [results[model]['ui'] for model in models if results[model].get('ui') is not None]
    
    if b_scores and ui_scores:
        avg_b = sum(b_scores) / len(b_scores)
        avg_ui = sum(ui_scores) / len(ui_scores)
        gap = avg_ui - avg_b
        
        print(f"   📏 Scaling Check:")
        print(f"     Avg B filter: {avg_b:.6f}")
        print(f"     Avg UI filter: {avg_ui:.6f}")
        print(f"     Performance gap: {gap:.6f}")
        
        if gap < 0.05:
            print(f"   ✅ Scaling: GOOD (gap < 0.05)")
        elif gap < 0.1:
            print(f"   ⚠️ Scaling: MODERATE (gap < 0.1)")
        else:
            print(f"   ❌ Scaling: POOR (gap > 0.1)")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Comprehensive UIB Filter Testing")
    parser.add_argument('--dataset', type=str, default='ml-100k', help='Dataset to test')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test (fewer filters)')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🏃 Quick UIB test mode")
        # Test only essential filters
        filters_to_test = ['ui', 'uib']
        models_to_test = ['enhanced']
    else:
        print("🔬 Comprehensive UIB test mode")
    
    start_time = time.time()
    results = test_comprehensive_uib()
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ UIB TESTING COMPLETED")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Results: {sum(len([v for v in model_results.values() if v is not None]) for model_results in results.values())} successful tests")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()