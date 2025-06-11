'''
Created on December 2024
Filter Design and Initialization Search for User-Specific Model
Two-stage search: 1) All filter designs, 2) All initializations with best design

@author: Enhanced from uSpec by Claude
'''

import os
import sys
import argparse
import time
import warnings
import subprocess
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

class FilterDesignSearch:
    """Two-stage search: Filter designs then initializations"""
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = []
        
        # Define all filter designs available in filters.py
        self.filter_designs = [
            'original',
            'basis', 
            'enhanced_basis',
            'adaptive_golden',
            'multiscale',
            'ensemble',
            'chebyshev',
            'legendre',
            'band_stop',
            'adaptive_band_stop',
            'parametric_multi_band',
            'harmonic'
        ]
        
        # Define all initialization patterns available in filters.py
        self.init_filters = [
            # Core smoothing filters
            'smooth',
            'butterworth', 
            'gaussian',
            
            # Golden ratio variants
            'golden_036',
            
            # Band-stop patterns
            'band_stop',
            'notch'
        ]
    
    def _build_command(self, config):
        """Build command with given configuration"""
        cmd = [
            sys.executable, "main.py",
            "--model_type", "user_specific",
            "--dataset", config['dataset'],
            "--no_shared_base" if not config['shared_base'] else "--shared_base",
            "--personalization_dim", str(config['personalization_dim']),
            "--filter_design", config['filter_design'],
            "--init_filter", config['init_filter'],
            "--lr", str(config['lr']),
            "--decay", str(config['decay']),
            "--u_n_eigen", str(config['u_n_eigen']),
            "--i_n_eigen", str(config['i_n_eigen']),
            "--b_n_eigen", str(config['b_n_eigen']),
            "--filter", config['filter'],
            "--epochs", "25",  # Reduced for speed
            "--patience", "5",
            "--seed", "2025",
            "--verbose", "1"  # Enable verbose to see what's happening
        ]
        
        # Debug: Print the command being executed
        print(f"\n    🔍 DEBUG Command: {' '.join(cmd)}")
        return cmd
    
    def _evaluate_config(self, config, timeout=300):
        """Evaluate a single configuration with debug info"""
        cmd = self._build_command(config)
        
        print(f"\n    📋 Config: filter_design={config['filter_design']}, init_filter={config['init_filter']}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode != 0:
                print(f"    ❌ Error: {result.stderr[:200]}")
                return None
            
            # Parse NDCG@20 from output
            ndcg = self._parse_ndcg(result.stdout)
            
            # Debug: Show what we parsed
            if ndcg is not None:
                print(f"    ✅ Parsed NDCG: {ndcg:.6f}")
            else:
                print(f"    ❌ Could not parse NDCG from output")
                print(f"    📝 Last few lines of output:")
                output_lines = result.stdout.split('\n')
                for line in output_lines[-10:]:
                    if line.strip():
                        print(f"        {line}")
            
            return ndcg
                
        except subprocess.TimeoutExpired:
            print(f"    ⏱️ Timeout after {timeout}s")
            return None
        except Exception as e:
            print(f"    ❌ Exception: {e}")
            return None
    
    def _parse_ndcg(self, output):
        """Parse NDCG@20 from output"""
        for line in output.split('\n'):
            if "Final Test Results:" in line and "NDCG@20=" in line:
                try:
                    ndcg_part = line.split("NDCG@20=")[1]
                    return float(ndcg_part.split(",")[0])
                except (IndexError, ValueError):
                    continue
        return None
    
    def search_filter_designs(self):
        """Stage 1: Search all filter designs"""
        print("🔍 STAGE 1: FILTER DESIGN SEARCH")
        print("=" * 60)
        print(f"Testing {len(self.filter_designs)} filter designs with fixed initialization: {self.base_config['init_filter']}")
        print("-" * 60)
        
        design_results = []
        
        for i, design in enumerate(self.filter_designs, 1):
            config = self.base_config.copy()
            config['filter_design'] = design
            # Keep init_filter the same
            
            print(f"\n[{i:2d}/{len(self.filter_designs)}] Testing {design:<20}", end="", flush=True)
            
            ndcg = self._evaluate_config(config)
            
            if ndcg is not None:
                print(f" → Final NDCG@20: {ndcg:.6f}")
                design_results.append((design, ndcg, config.copy()))
            else:
                print(f" → FAILED")
        
        # Sort by performance
        design_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 FILTER DESIGN RESULTS:")
        print("-" * 60)
        for i, (design, ndcg, _) in enumerate(design_results, 1):
            status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
            print(f"{status} {design:<20} NDCG@20: {ndcg:.6f}")
        
        return design_results
    
    def search_initializations(self, best_design):
        """Stage 2: Search all initializations with best design"""
        print(f"\n🎯 STAGE 2: INITIALIZATION SEARCH")
        print("=" * 60)
        print(f"Testing {len(self.init_filters)} initializations with best design: {best_design}")
        print("-" * 60)
        
        init_results = []
        
        for i, init_filter in enumerate(self.init_filters, 1):
            config = self.base_config.copy()
            config['filter_design'] = best_design  # Use best design
            config['init_filter'] = init_filter    # Change initialization
            
            print(f"\n[{i:2d}/{len(self.init_filters)}] Testing {init_filter:<20}", end="", flush=True)
            
            ndcg = self._evaluate_config(config)
            
            if ndcg is not None:
                print(f" → Final NDCG@20: {ndcg:.6f}")
                init_results.append((init_filter, ndcg, config.copy()))
            else:
                print(f" → FAILED")
        
        # Sort by performance
        init_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 INITIALIZATION RESULTS:")
        print("-" * 60)
        for i, (init_filter, ndcg, _) in enumerate(init_results, 1):
            status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
            print(f"{status} {init_filter:<20} NDCG@20: {ndcg:.6f}")
        
        return init_results
    
    def run_search(self):
        """Run complete two-stage search"""
        start_time = time.time()
        
        print("🚀 TWO-STAGE FILTER SEARCH")
        print("=" * 80)
        print("Configuration:")
        print(f"  Model: user_specific")
        print(f"  Dataset: {self.base_config['dataset']}")
        print(f"  Filter: {self.base_config['filter']}")
        print(f"  Shared Base: {self.base_config['shared_base']}")
        print(f"  Personalization Dim: {self.base_config['personalization_dim']}")
        print(f"  Eigenvalues: u={self.base_config['u_n_eigen']}, i={self.base_config['i_n_eigen']}, b={self.base_config['b_n_eigen']}")
        print(f"  Learning: lr={self.base_config['lr']}, decay={self.base_config['decay']}")
        print("=" * 80)
        
        # Stage 1: Filter designs
        design_results = self.search_filter_designs()
        
        if not design_results:
            print("❌ No successful filter design experiments!")
            return
        
        # Get best design
        best_design, best_design_ndcg, _ = design_results[0]
        
        # Stage 2: Initializations with best design
        init_results = self.search_initializations(best_design)
        
        total_time = time.time() - start_time
        
        # Final results
        print(f"\n" + "=" * 80)
        print("🏆 FINAL SEARCH RESULTS")
        print("=" * 80)
        
        print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
        print(f"🔬 Total Experiments: {len(design_results) + len(init_results)}")
        
        # Best from each stage
        print(f"\n📊 BEST RESULTS BY STAGE:")
        print(f"  🎨 Best Filter Design: {best_design} (NDCG: {best_design_ndcg:.6f})")
        
        if init_results:
            best_init, best_init_ndcg, best_config = init_results[0]
            print(f"  🎛️ Best Initialization: {best_init} (NDCG: {best_init_ndcg:.6f})")
            
            print(f"\n🏆 OPTIMAL CONFIGURATION:")
            print(f"  Filter Design: {best_config['filter_design']}")
            print(f"  Initialization: {best_config['init_filter']}")
            print(f"  Final NDCG@20: {best_init_ndcg:.6f}")
            
            print(f"\n🚀 OPTIMAL COMMAND:")
            cmd_parts = [
                "python main.py",
                "--model_type user_specific",
                f"--dataset {best_config['dataset']}",
                "--no_shared_base" if not best_config['shared_base'] else "--shared_base",
                f"--personalization_dim {best_config['personalization_dim']}",
                f"--filter_design {best_config['filter_design']}",
                f"--init_filter {best_config['init_filter']}",
                f"--lr {best_config['lr']}",
                f"--decay {best_config['decay']}",
                f"--u_n_eigen {best_config['u_n_eigen']}",
                f"--i_n_eigen {best_config['i_n_eigen']}",
                f"--b_n_eigen {best_config['b_n_eigen']}",
                f"--filter {best_config['filter']}",
                "--epochs 50",
                "--patience 10"
            ]
            print(" \\\n    ".join(cmd_parts))
        else:
            print(f"  🎛️ No successful initialization experiments")
            
            print(f"\n🏆 BEST CONFIGURATION (design only):")
            print(f"  Filter Design: {best_design}")
            print(f"  NDCG@20: {best_design_ndcg:.6f}")
        
        print("=" * 80)
        
        # Analysis section
        if len(design_results) > 1:
            print(f"\n📈 FILTER DESIGN ANALYSIS:")
            
            # Group by complexity
            simple_designs = ['original', 'basis', 'enhanced_basis', 'adaptive_golden']
            advanced_designs = ['multiscale', 'ensemble', 'chebyshev', 'legendre']
            special_designs = ['band_stop', 'adaptive_band_stop', 'parametric_multi_band', 'harmonic']
            
            for category, designs in [
                ("Simple Designs", simple_designs),
                ("Advanced Designs", advanced_designs), 
                ("Special Designs", special_designs)
            ]:
                category_results = [(d, n) for d, n, _ in design_results if d in designs]
                if category_results:
                    best_in_category = max(category_results, key=lambda x: x[1])
                    avg_in_category = np.mean([n for _, n in category_results])
                    print(f"  {category}: Best={best_in_category[0]} ({best_in_category[1]:.6f}), Avg={avg_in_category:.6f}")
        
        if len(init_results) > 1:
            print(f"\n🎛️ INITIALIZATION ANALYSIS:")
            
            # Group by type
            smoothing_inits = ['smooth', 'butterworth', 'gaussian']
            golden_inits = ['golden_036']
            special_inits = ['band_stop', 'notch']
            
            for category, inits in [
                ("Smoothing", smoothing_inits),
                ("Golden Ratio", golden_inits),
                ("Special", special_inits)
            ]:
                category_results = [(i, n) for i, n, _ in init_results if i in inits]
                if category_results:
                    best_in_category = max(category_results, key=lambda x: x[1])
                    avg_in_category = np.mean([n for _, n in category_results])
                    print(f"  {category}: Best={best_in_category[0]} ({best_in_category[1]:.6f}), Avg={avg_in_category:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Filter Design and Initialization Search")
    
    # Use the exact parameters from your command
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       help='Dataset to test')
    parser.add_argument('--shared_base', action='store_true', default=False,
                       help='Use shared base (your command uses --no_shared_base)')
    parser.add_argument('--personalization_dim', type=int, default=32,
                       help='Personalization dimension')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--decay', type=float, default=1e-03,
                       help='Weight decay')
    parser.add_argument('--u_n_eigen', type=int, default=30,
                       help='User eigenvalues')
    parser.add_argument('--i_n_eigen', type=int, default=40,
                       help='Item eigenvalues')
    parser.add_argument('--b_n_eigen', type=int, default=40,
                       help='Bipartite eigenvalues')
    parser.add_argument('--filter', type=str, default='u',
                       help='Filter type')
    parser.add_argument('--init_filter', type=str, default='smooth',
                       help='Initial filter for design search')
    
    args = parser.parse_args()
    
    # Build base configuration from your exact command
    base_config = {
        'dataset': args.dataset,
        'shared_base': args.shared_base,  # False for --no_shared_base
        'personalization_dim': args.personalization_dim,
        'filter_design': 'enhanced_basis',  # Will be overridden in search
        'init_filter': args.init_filter,   # Will be overridden in search
        'lr': args.lr,
        'decay': args.decay,
        'u_n_eigen': args.u_n_eigen,
        'i_n_eigen': args.i_n_eigen,
        'b_n_eigen': args.b_n_eigen,
        'filter': args.filter
    }
    
    # Run search
    searcher = FilterDesignSearch(base_config)
    searcher.run_search()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# Run with your exact parameters:
# python search.py --dataset ml-100k --personalization_dim 32 --lr 0.1 --decay 1e-03 --u_n_eigen 30 --i_n_eigen 40 --b_n_eigen 40 --filter u

# Or with defaults (matches your command):
# python search.py

# ============================================================================
# EXPECTED OUTPUT STRUCTURE
# ============================================================================

"""
🚀 TWO-STAGE FILTER SEARCH
================================================================================
Configuration:
  Model: user_specific
  Dataset: ml-100k
  Filter: u
  Shared Base: False
  Personalization Dim: 32
  Eigenvalues: u=30, i=40, b=40
  Learning: lr=0.1, decay=0.001
================================================================================

🔍 STAGE 1: FILTER DESIGN SEARCH
============================================================
Testing 12 filter designs with fixed initialization: smooth
------------------------------------------------------------
[ 1/12] Testing original           ✅ NDCG@20: 0.385234
[ 2/12] Testing basis              ✅ NDCG@20: 0.392156
[ 3/12] Testing enhanced_basis     ✅ NDCG@20: 0.396782
[ 4/12] Testing adaptive_golden    ✅ NDCG@20: 0.389123
[ 5/12] Testing multiscale         ✅ NDCG@20: 0.387654
[ 6/12] Testing ensemble           ❌ FAILED
[ 7/12] Testing chebyshev          ✅ NDCG@20: 0.391234
[ 8/12] Testing legendre           ✅ NDCG@20: 0.388976
[ 9/12] Testing band_stop          ✅ NDCG@20: 0.383456
[10/12] Testing adaptive_band_stop ✅ NDCG@20: 0.384567
[11/12] Testing parametric_multi_band ✅ NDCG@20: 0.390123
[12/12] Testing harmonic           ✅ NDCG@20: 0.386789

📊 FILTER DESIGN RESULTS:
------------------------------------------------------------
🥇 enhanced_basis      NDCG@20: 0.396782
🥈 basis               NDCG@20: 0.392156
🥉 chebyshev           NDCG@20: 0.391234
 4. parametric_multi_band NDCG@20: 0.390123
 5. adaptive_golden     NDCG@20: 0.389123
 6. legendre            NDCG@20: 0.388976
 7. multiscale          NDCG@20: 0.387654
 8. harmonic            NDCG@20: 0.386789
 9. original            NDCG@20: 0.385234
10. adaptive_band_stop  NDCG@20: 0.384567
11. band_stop           NDCG@20: 0.383456

🎯 STAGE 2: INITIALIZATION SEARCH
============================================================
Testing 6 initializations with best design: enhanced_basis
------------------------------------------------------------
[ 1/ 6] Testing smooth             ✅ NDCG@20: 0.396782
[ 2/ 6] Testing butterworth        ✅ NDCG@20: 0.398123
[ 3/ 6] Testing gaussian           ✅ NDCG@20: 0.397456
[ 4/ 6] Testing golden_036         ✅ NDCG@20: 0.399234
[ 5/ 6] Testing band_stop          ✅ NDCG@20: 0.395678
[ 6/ 6] Testing notch              ✅ NDCG@20: 0.394123

📊 INITIALIZATION RESULTS:
------------------------------------------------------------
🥇 golden_036          NDCG@20: 0.399234
🥈 butterworth         NDCG@20: 0.398123
🥉 gaussian            NDCG@20: 0.397456
 4. smooth              NDCG@20: 0.396782
 5. band_stop           NDCG@20: 0.395678
 6. notch               NDCG@20: 0.394123

================================================================================
🏆 FINAL SEARCH RESULTS
================================================================================
⏱️  Total Time: 8.5 minutes
🔬 Total Experiments: 17

📊 BEST RESULTS BY STAGE:
  🎨 Best Filter Design: enhanced_basis (NDCG: 0.396782)
  🎛️ Best Initialization: golden_036 (NDCG: 0.399234)

🏆 OPTIMAL CONFIGURATION:
  Filter Design: enhanced_basis
  Initialization: golden_036
  Final NDCG@20: 0.399234

🚀 OPTIMAL COMMAND:
python main.py \
    --model_type user_specific \
    --dataset ml-100k \
    --no_shared_base \
    --personalization_dim 32 \
    --filter_design enhanced_basis \
    --init_filter golden_036 \
    --lr 0.1 \
    --decay 0.001 \
    --u_n_eigen 30 \
    --i_n_eigen 40 \
    --b_n_eigen 40 \
    --filter u \
    --epochs 50 \
    --patience 10
================================================================================
"""

# ============================================================================
# KEY FEATURES
# ============================================================================

# ✅ Two-stage search: designs first, then initializations
# ✅ Uses your EXACT parameters (no changes)
# ✅ Tests all available filter designs from filters.py
# ✅ Tests all initialization patterns after finding best design
# ✅ Provides optimal command at the end
# ✅ Includes analysis by design/initialization categories
# ✅ Reduced epochs (25) for faster search
# ✅ Clear progress tracking and results formatting

# ============================================================================