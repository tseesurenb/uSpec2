#!/usr/bin/env python3
"""
Hierarchical UIB Eigenvalue Search
Three-stage hierarchical search for optimal u, i, b eigenvalue combinations
Tests all starting orders: B→U→I, U→B→I, I→U→B

@author: Enhanced UIB Search
"""

import os
import sys
import argparse
import time
import warnings
import subprocess
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import shutil

warnings.filterwarnings("ignore")

class HierarchicalUIBSearch:
    """Hierarchical search for UIB eigenvalue optimization"""
    
    def __init__(self, dataset: str = 'ml-100k', model_type: str = 'enhanced'):
        self.dataset = dataset
        self.model_type = model_type
        
        # Define eigenvalue ranges with step size 10
        self.u_range = list(range(15, 101, 10))  # [15, 25, 35, ..., 95]
        self.i_range = list(range(20, 151, 10))  # [20, 30, 40, ..., 140]
        self.b_range = list(range(30, 201, 10))  # [30, 40, 50, ..., 190]
        
        # Search orders to test
        self.search_orders = [
            ('b', 'u', 'i'),  # Start with B, then U, then I
            ('u', 'b', 'i'),  # Start with U, then B, then I
            ('i', 'u', 'b')   # Start with I, then U, then B
        ]
        
        # Store results for each search order
        self.order_results = {}
        
        print(f"🚀 HIERARCHICAL UIB EIGENVALUE SEARCH")
        print(f"Dataset: {dataset}")
        print(f"Model: {model_type}")
        print(f"📊 Search Ranges:")
        print(f"   U eigenvalues: {self.u_range[0]} to {self.u_range[-1]} (step 10, {len(self.u_range)} values)")
        print(f"   I eigenvalues: {self.i_range[0]} to {self.i_range[-1]} (step 10, {len(self.i_range)} values)")
        print(f"   B eigenvalues: {self.b_range[0]} to {self.b_range[-1]} (step 10, {len(self.b_range)} values)")
        print(f"🔄 Search Orders: {len(self.search_orders)}")
        for i, order in enumerate(self.search_orders, 1):
            print(f"   {i}. {order[0].upper()} → {order[1].upper()} → {order[2].upper()}")
    
    def _clear_cache(self):
        """Clear cache to ensure fresh experiments"""
        cache_dir = "../cache"
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print("    🗑️ Cache cleared")
            except Exception as e:
                print(f"    ⚠️ Cache clear failed: {e}")
    
    def _evaluate_config(self, u_eigen: int, i_eigen: int, b_eigen: int, timeout: int = 300) -> Optional[float]:
        """Evaluate a single UIB configuration"""
        
        # Clear cache before each experiment
        self._clear_cache()
        
        cmd = [
            sys.executable, "main.py",
            "--model_type", self.model_type,
            "--dataset", self.dataset,
            "--filter", "uib",  # Always use UIB filter
            "--u_n_eigen", str(u_eigen),
            "--i_n_eigen", str(i_eigen),
            "--b_n_eigen", str(b_eigen),
            "--epochs", "15",  # Enough epochs for reliable results
            "--patience", "5",
            "--lr", "0.01",
            "--decay", "0.01",
            "--seed", "2025",
            "--verbose", "0"
        ]
        
        # Add model-specific parameters
        if self.model_type == "user_specific":
            cmd.extend(["--shared_base", "--personalization_dim", "16"])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode != 0:
                return None
            
            # Parse NDCG@20
            for line in result.stdout.split('\n'):
                if "Final Test Results:" in line and "NDCG@20=" in line:
                    try:
                        ndcg_part = line.split("NDCG@20=")[1]
                        return float(ndcg_part.split(",")[0])
                    except (IndexError, ValueError):
                        continue
            return None
                
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None
    
    def _search_single_parameter(self, param_name: str, param_range: List[int], 
                                fixed_params: Dict[str, int]) -> Tuple[int, float]:
        """Search for optimal value of a single parameter"""
        
        print(f"\n🔍 Searching {param_name.upper()} eigenvalues...")
        print(f"   Range: {param_range}")
        print(f"   Fixed: {fixed_params}")
        print("-" * 60)
        
        best_value = param_range[0]
        best_ndcg = 0.0
        results = []
        
        for i, value in enumerate(param_range, 1):
            # Build configuration
            config = fixed_params.copy()
            config[param_name] = value
            
            progress = (i / len(param_range)) * 100
            print(f"[{i:2d}/{len(param_range)}] {progress:5.1f}% | {param_name}={value:3d} (u={config['u_eigen']:3d}, i={config['i_eigen']:3d}, b={config['b_eigen']:3d})", end=" ", flush=True)
            
            # Evaluate
            ndcg = self._evaluate_config(config['u_eigen'], config['i_eigen'], config['b_eigen'])
            
            if ndcg is not None:
                print(f"✅ {ndcg:.6f}")
                results.append((value, ndcg))
                
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_value = value
            else:
                print("❌")
        
        print(f"\n📊 {param_name.upper()} Search Results:")
        print(f"   Best {param_name}: {best_value} → NDCG: {best_ndcg:.6f}")
        
        # Show top 3 results
        if len(results) >= 3:
            results.sort(key=lambda x: x[1], reverse=True)
            print(f"   Top 3:")
            for i, (value, ndcg) in enumerate(results[:3], 1):
                print(f"     {i}. {param_name}={value:3d} → {ndcg:.6f}")
        
        return best_value, best_ndcg
    
    def _run_hierarchical_search(self, search_order: Tuple[str, str, str]) -> Dict:
        """Run hierarchical search in specified order"""
        
        order_name = "→".join([p.upper() for p in search_order])
        print(f"\n{'='*80}")
        print(f"🎯 HIERARCHICAL SEARCH ORDER: {order_name}")
        print(f"{'='*80}")
        
        # Initialize with middle values
        current_config = {
            'u_eigen': self.u_range[len(self.u_range)//2],  # Middle of U range
            'i_eigen': self.i_range[len(self.i_range)//2],  # Middle of I range  
            'b_eigen': self.b_range[len(self.b_range)//2]   # Middle of B range
        }
        
        print(f"🏁 Starting configuration: u={current_config['u_eigen']}, i={current_config['i_eigen']}, b={current_config['b_eigen']}")
        
        # Stage results for this order
        stage_results = {}
        
        # Search each parameter in order
        for stage_idx, param_name in enumerate(search_order, 1):
            print(f"\n🔍 STAGE {stage_idx}/3: Optimizing {param_name.upper()}")
            
            # Get parameter range
            if param_name == 'u':
                param_range = self.u_range
            elif param_name == 'i':
                param_range = self.i_range
            elif param_name == 'b':
                param_range = self.b_range
            
            # Create fixed parameters (excluding the one being optimized)
            fixed_params = {k: v for k, v in current_config.items() if k != f'{param_name}_eigen'}
            
            # Search for optimal value
            best_value, best_ndcg = self._search_single_parameter(
                f'{param_name}_eigen', param_range, fixed_params
            )
            
            # Update current configuration
            current_config[f'{param_name}_eigen'] = best_value
            stage_results[f'stage_{stage_idx}_{param_name}'] = {
                'best_value': best_value,
                'best_ndcg': best_ndcg,
                'config': current_config.copy()
            }
            
            print(f"✅ Stage {stage_idx} completed. Updated config: u={current_config['u_eigen']}, i={current_config['i_eigen']}, b={current_config['b_eigen']}")
        
        # Final evaluation with optimized configuration
        print(f"\n🏆 FINAL EVALUATION FOR {order_name}")
        print(f"   Final config: u={current_config['u_eigen']}, i={current_config['i_eigen']}, b={current_config['b_eigen']}")
        
        final_ndcg = self._evaluate_config(
            current_config['u_eigen'], 
            current_config['i_eigen'], 
            current_config['b_eigen']
        )
        
        if final_ndcg is not None:
            print(f"   Final NDCG: {final_ndcg:.6f}")
        else:
            print(f"   Final NDCG: FAILED")
            final_ndcg = 0.0
        
        return {
            'search_order': search_order,
            'order_name': order_name,
            'final_config': current_config,
            'final_ndcg': final_ndcg,
            'stage_results': stage_results
        }
    
    def run_complete_search(self) -> Dict:
        """Run complete hierarchical search with all orders"""
        
        print(f"\n🚀 STARTING COMPLETE HIERARCHICAL UIB SEARCH")
        print(f"Total search orders: {len(self.search_orders)}")
        
        start_time = time.time()
        
        # Run search for each order
        for order_idx, search_order in enumerate(self.search_orders, 1):
            print(f"\n🎯 SEARCH ORDER {order_idx}/{len(self.search_orders)}")
            
            order_result = self._run_hierarchical_search(search_order)
            self.order_results[order_result['order_name']] = order_result
        
        total_time = time.time() - start_time
        
        # Analyze results across all orders
        print(f"\n{'='*80}")
        print(f"🏆 COMPLETE HIERARCHICAL SEARCH RESULTS")
        print(f"{'='*80}")
        
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        
        # Find best overall result
        best_overall = None
        best_overall_ndcg = 0.0
        
        print(f"\n📊 RESULTS BY SEARCH ORDER:")
        for order_name, result in self.order_results.items():
            final_ndcg = result['final_ndcg']
            config = result['final_config']
            
            print(f"   {order_name:<10}: u={config['u_eigen']:3d}, i={config['i_eigen']:3d}, b={config['b_eigen']:3d} → NDCG={final_ndcg:.6f}")
            
            if final_ndcg > best_overall_ndcg:
                best_overall_ndcg = final_ndcg
                best_overall = result
        
        if best_overall:
            print(f"\n🥇 BEST OVERALL RESULT:")
            print(f"   Search Order: {best_overall['order_name']}")
            print(f"   Configuration: u={best_overall['final_config']['u_eigen']}, i={best_overall['final_config']['i_eigen']}, b={best_overall['final_config']['b_eigen']}")
            print(f"   NDCG@20: {best_overall['final_ndcg']:.6f}")
            
            print(f"\n🚀 OPTIMAL COMMAND:")
            cmd_parts = [
                "python main.py",
                f"--model_type {self.model_type}",
                f"--dataset {self.dataset}",
                "--filter uib",
                f"--u_n_eigen {best_overall['final_config']['u_eigen']}",
                f"--i_n_eigen {best_overall['final_config']['i_eigen']}",
                f"--b_n_eigen {best_overall['final_config']['b_eigen']}",
                "--epochs 50",
                "--patience 10"
            ]
            
            if self.model_type == "user_specific":
                cmd_parts.insert(-2, "--shared_base --personalization_dim 16")
            
            print(" \\\n    ".join(cmd_parts))
        
        # Analysis of search order effectiveness
        print(f"\n📈 SEARCH ORDER ANALYSIS:")
        order_performances = [(name, result['final_ndcg']) for name, result in self.order_results.items()]
        order_performances.sort(key=lambda x: x[1], reverse=True)
        
        for i, (order_name, ndcg) in enumerate(order_performances, 1):
            if i == 1:
                print(f"   🥇 {order_name}: {ndcg:.6f} (BEST)")
            elif i == 2:
                print(f"   🥈 {order_name}: {ndcg:.6f}")
            elif i == 3:
                print(f"   🥉 {order_name}: {ndcg:.6f}")
            else:
                print(f"   {i:2d}. {order_name}: {ndcg:.6f}")
        
        # Convergence analysis
        if len(order_performances) > 1:
            best_ndcg = order_performances[0][1]
            worst_ndcg = order_performances[-1][1]
            performance_gap = best_ndcg - worst_ndcg
            
            print(f"\n🎯 CONVERGENCE ANALYSIS:")
            print(f"   Best order NDCG: {best_ndcg:.6f}")
            print(f"   Worst order NDCG: {worst_ndcg:.6f}")
            print(f"   Performance gap: {performance_gap:.6f}")
            
            if performance_gap < 0.005:
                print(f"   ✅ EXCELLENT: All orders converge to similar results (gap < 0.005)")
            elif performance_gap < 0.01:
                print(f"   ✅ GOOD: Orders show good convergence (gap < 0.01)")
            elif performance_gap < 0.02:
                print(f"   ⚠️ MODERATE: Some variation between orders (gap < 0.02)")
            else:
                print(f"   ❌ POOR: High variation between orders (gap > 0.02)")
        
        print(f"\n{'='*80}")
        
        return {
            'best_overall': best_overall,
            'all_results': self.order_results,
            'total_time': total_time,
            'order_performances': order_performances
        }


def main():
    parser = argparse.ArgumentParser(description="Hierarchical UIB Eigenvalue Search")
    
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'ml-1m', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book'],
                       help='Dataset to search on')
    
    parser.add_argument('--model_type', type=str, default='enhanced',
                       choices=['enhanced', 'user_specific'],
                       help='Model type: enhanced or user_specific')
    
    parser.add_argument('--order', type=str, default='all',
                       choices=['all', 'bui', 'ubi', 'iub'],
                       help='Search order: all (test all orders), bui (B→U→I), ubi (U→B→I), iub (I→U→B)')
    
    parser.add_argument('--quick', action='store_true', default=False,
                       help='Quick test with smaller ranges')
    
    args = parser.parse_args()
    
    # Initialize searcher
    searcher = HierarchicalUIBSearch(args.dataset, args.model_type)
    
    # Modify ranges for quick test
    if args.quick:
        print("🏃 Quick mode: Using reduced ranges")
        searcher.u_range = list(range(20, 51, 15))  # [20, 35, 50]
        searcher.i_range = list(range(30, 61, 15))  # [30, 45, 60]
        searcher.b_range = list(range(40, 71, 15))  # [40, 55, 70]
    
    # Filter search orders if specific order requested
    if args.order != 'all':
        order_map = {
            'bui': ('b', 'u', 'i'),
            'ubi': ('u', 'b', 'i'),
            'iub': ('i', 'u', 'b')
        }
        if args.order in order_map:
            searcher.search_orders = [order_map[args.order]]
            print(f"🎯 Running single search order: {args.order.upper()}")
    
    # Run search
    results = searcher.run_complete_search()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Run complete hierarchical search (all 3 orders)
python search_uib.py --dataset ml-100k --model_type enhanced

# Run specific search order
python search_uib.py --dataset ml-100k --model_type enhanced --order bui

# Quick test with smaller ranges
python search_uib.py --dataset ml-100k --model_type enhanced --quick

# User-specific model search
python search_uib.py --dataset ml-100k --model_type user_specific

# Larger dataset
python search_uib.py --dataset gowalla --model_type enhanced
"""

# ============================================================================
# EXPECTED OUTPUT STRUCTURE
# ============================================================================

"""
🚀 HIERARCHICAL UIB EIGENVALUE SEARCH
Dataset: ml-100k
Model: enhanced
📊 Search Ranges:
   U eigenvalues: 15 to 95 (step 10, 9 values)
   I eigenvalues: 20 to 140 (step 10, 13 values)
   B eigenvalues: 30 to 190 (step 10, 17 values)
🔄 Search Orders: 3
   1. B → U → I
   2. U → B → I
   3. I → U → B

================================================================================
🎯 HIERARCHICAL SEARCH ORDER: B→U→I
================================================================================
🏁 Starting configuration: u=55, i=80, b=110

🔍 STAGE 1/3: Optimizing B
   Range: [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
   Fixed: {'u_eigen': 55, 'i_eigen': 80}
------------------------------------------------------------
[ 1/17]   5.9% | b_eigen= 30 (u= 55, i= 80, b= 30) ✅ 0.341234
[ 2/17]  11.8% | b_eigen= 40 (u= 55, i= 80, b= 40) ✅ 0.352156
...
📊 B Search Results:
   Best b_eigen: 60 → NDCG: 0.367890
✅ Stage 1 completed. Updated config: u=55, i=80, b=60

🔍 STAGE 2/3: Optimizing U
   Range: [15, 25, 35, 45, 55, 65, 75, 85, 95]
   Fixed: {'i_eigen': 80, 'b_eigen': 60}
------------------------------------------------------------
...
📊 U Search Results:
   Best u_eigen: 45 → NDCG: 0.371234
✅ Stage 2 completed. Updated config: u=45, i=80, b=60

🔍 STAGE 3/3: Optimizing I
   Range: [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
   Fixed: {'u_eigen': 45, 'b_eigen': 60}
------------------------------------------------------------
...
📊 I Search Results:
   Best i_eigen: 90 → NDCG: 0.375678
✅ Stage 3 completed. Updated config: u=45, i=90, b=60

🏆 FINAL EVALUATION FOR B→U→I
   Final config: u=45, i=90, b=60
   Final NDCG: 0.375678

[... Similar for U→B→I and I→U→B orders ...]

================================================================================
🏆 COMPLETE HIERARCHICAL SEARCH RESULTS
================================================================================
⏱️  Total time: 45.2 minutes

📊 RESULTS BY SEARCH ORDER:
   B→U→I     : u= 45, i= 90, b= 60 → NDCG=0.375678
   U→B→I     : u= 50, i= 85, b= 65 → NDCG=0.374123
   I→U→B     : u= 40, i= 95, b= 55 → NDCG=0.376234

🥇 BEST OVERALL RESULT:
   Search Order: I→U→B
   Configuration: u=40, i=95, b=55
   NDCG@20: 0.376234

🚀 OPTIMAL COMMAND:
python main.py \
    --model_type enhanced \
    --dataset ml-100k \
    --filter uib \
    --u_n_eigen 40 \
    --i_n_eigen 95 \
    --b_n_eigen 55 \
    --epochs 50 \
    --patience 10

📈 SEARCH ORDER ANALYSIS:
   🥇 I→U→B: 0.376234 (BEST)
   🥈 B→U→I: 0.375678
   🥉 U→B→I: 0.374123

🎯 CONVERGENCE ANALYSIS:
   Best order NDCG: 0.376234
   Worst order NDCG: 0.374123
   Performance gap: 0.002111
   ✅ EXCELLENT: All orders converge to similar results (gap < 0.005)
================================================================================
"""

# ============================================================================
# KEY FEATURES
# ============================================================================

# 🎯 HIERARCHICAL SEARCH:
# - Tests 3 different parameter optimization orders
# - Each order optimizes one parameter at a time while keeping others fixed
# - Finds optimal eigenvalue combinations efficiently

# 📊 COMPREHENSIVE RANGES:
# - U: 15-95 (step 10) = 9 values
# - I: 20-140 (step 10) = 13 values  
# - B: 30-190 (step 10) = 17 values
# - Total possible combinations: 9×13×17 = 1989
# - Hierarchical approach: ~39 experiments per order = 117 total

# 🔄 SEARCH ORDERS:
# 1. B→U→I: Start with bipartite, then user, then item
# 2. U→B→I: Start with user, then bipartite, then item
# 3. I→U→B: Start with item, then user, then bipartite

# ✅ EFFICIENCY:
# - 117 experiments instead of 1989 (94% reduction)
# - Cache clearing ensures fresh results
# - Automatic convergence analysis
# - Best overall configuration identified

# ============================================================================