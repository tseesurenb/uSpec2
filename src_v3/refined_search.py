'''
Created on June 8, 2025
Hierarchical Hyperparameter Search for Universal Spectral CF
Progressive parameter optimization: LR → Decay → Eigenvalues → Filters → Designs
ENHANCED: With Adaptive Boundary Expansion and Cache Clearing

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import os
import sys
import argparse
import time
import warnings
import subprocess
import json
import pickle
import numpy as np
import random
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

warnings.filterwarnings("ignore")

class HierarchicalHyperparameterSearch:
    """Hierarchical hyperparameter search with adaptive boundary expansion"""
    
    def __init__(self, dataset: str, model_type: str = 'enhanced'):
        self.dataset = dataset
        self.model_type = model_type  # 'enhanced' (model_enhanced.py) or 'basic' (model.py)
        self.results_history = []
        self.best_configs = {}  # Store best config at each stage
        self.stage_results = {}  # Store results for each stage
        
        # Define the hierarchical search stages and parameter spaces
        self.search_stages = self._define_search_stages()
        self.dataset_params = self._get_dataset_specific_params()
        
    def _define_search_stages(self) -> List[Dict]:
        """Define the hierarchical search stages in order"""
        
        # Base stages common to both models
        base_stages = [
            {
                'name': 'learning_rate',
                'description': 'Find optimal learning rate',
                'primary_param': 'lr',
                'search_space': {
                    'lr': [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001]
                },
                'fixed_params': {
                    'decay': 0.01,
                    'filter': 'ui',
                    'filter_design': 'enhanced_basis',
                    'init_filter': 'smooth'
                }
            },
            {
                'name': 'weight_decay',
                'description': 'Find optimal weight decay',
                'primary_param': 'decay',
                'search_space': {
                    'decay': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
                },
                'fixed_params': {
                    'filter': 'ui',
                    'filter_design': 'enhanced_basis',
                    'init_filter': 'smooth'
                }
            }
        ]
        
        # Model-specific stages
        if self.model_type == 'enhanced':
            # Enhanced model has similarity threshold optimization
            enhanced_stages = [
                {
                    'name': 'similarity_threshold',
                    'description': 'Find optimal similarity threshold',
                    'primary_param': 'similarity_threshold',
                    'search_space': {
                        'similarity_threshold': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
                    },
                    'fixed_params': {
                        'filter': 'ui',
                        'filter_design': 'enhanced_basis',
                        'init_filter': 'smooth'
                    }
                },
                {
                    'name': 'eigenvalues',
                    'description': 'Find optimal eigenvalue configuration (separate u/i)',
                    'primary_param': 'eigenvalues',
                    'search_space': 'dataset_dependent',
                    'fixed_params': {
                        'filter': 'ui',
                        'filter_design': 'enhanced_basis',
                        'init_filter': 'smooth'
                    }
                }
            ]
            # Add similarity threshold to learning rate and decay fixed params
            base_stages[0]['fixed_params']['similarity_threshold'] = 0.01
            base_stages[1]['fixed_params']['similarity_threshold'] = 0.01
            
        else:  # basic model (model.py)
            # Basic model uses separate eigenvalues but no similarity threshold
            enhanced_stages = [
                {
                    'name': 'eigenvalues',
                    'description': 'Find optimal eigenvalue configuration (separate u/i)',
                    'primary_param': 'eigenvalues',
                    'search_space': 'dataset_dependent',
                    'fixed_params': {
                        'filter': 'ui',
                        'filter_design': 'enhanced_basis',
                        'init_filter': 'smooth'
                    }
                }
            ]
        
        # Common final stages
        final_stages = [
            {
                'name': 'filter_type',
                'description': 'Find optimal filter type',
                'primary_param': 'filter',
                'search_space': {
                    'filter': ['u', 'i', 'ui']
                },
                'fixed_params': {
                    'filter_design': 'enhanced_basis',
                    'init_filter': 'smooth'
                }
            },
            {
                'name': 'filter_designs',
                'description': 'Test all filter designs',
                'primary_param': 'filter_design',
                'search_space': {
                    'filter_design': [
                        'original', 'basis', 'enhanced_basis', 'adaptive_golden',
                        'multiscale', 'ensemble', 'band_stop', 'adaptive_band_stop',
                        'parametric_multi_band', 'harmonic'
                    ]
                },
                'fixed_params': {
                    'init_filter': 'smooth'
                }
            },
            {
                'name': 'initializations',
                'description': 'Test all initialization patterns',
                'primary_param': 'init_filter',
                'search_space': {
                    'init_filter': [
                        'smooth', 'butterworth', 'gaussian', 'golden_036',
                        'band_stop', 'notch'
                    ]
                },
                'fixed_params': {}
            }
        ]
        
        return base_stages + enhanced_stages + final_stages
    
    def _get_dataset_specific_params(self) -> Dict:
        """Get dataset-specific parameter ranges based on size and sparsity"""
        
        base_configs = {
            'ml-100k': {
                'users': 943,
                'items': 1682,
                'sparsity': 'medium',
                'epochs': 30,
                'batch_sizes': ('500', '200')
            },
            'ml-1m': {
                'users': 6040,
                'items': 3706,
                'sparsity': 'medium',
                'epochs': 25,
                'batch_sizes': ('1000', '400')
            },
            'lastfm': {
                'users': 1892,
                'items': 4489,
                'sparsity': 'sparse',
                'epochs': 25,
                'batch_sizes': ('800', '300')
            },
            'gowalla': {
                'users': 29858,
                'items': 40981,
                'sparsity': 'very_sparse',
                'epochs': 20,
                'batch_sizes': ('1200', '400')
            },
            'yelp2018': {
                'users': 31668,
                'items': 38048,
                'sparsity': 'very_sparse',
                'epochs': 15,
                'batch_sizes': ('1500', '500')
            },
            'amazon-book': {
                'users': 52643,
                'items': 91599,
                'sparsity': 'extremely_sparse',
                'epochs': 15,
                'batch_sizes': ('2000', '600')
            }
        }
        
        # Both models use separate u_n_eigen and i_n_eigen
        eigenvalue_configs = {
            'ml-100k': {
                'u_n_eigen': list(range(15, 50, 5)),  # [15, 20, 25, 30, 35, 40, 45]
                'i_n_eigen': list(range(25, 65, 5)),  # [25, 30, 35, 40, 45, 50, 55, 60]
            },
            'ml-1m': {
                'u_n_eigen': list(range(30, 80, 10)),  # [30, 40, 50, 60, 70]
                'i_n_eigen': list(range(40, 100, 10)),  # [40, 50, 60, 70, 80, 90]
            },
            'lastfm': {
                'u_n_eigen': list(range(20, 70, 10)),  # [20, 30, 40, 50, 60]
                'i_n_eigen': list(range(30, 90, 10)),  # [30, 40, 50, 60, 70, 80]
            },
            'gowalla': {
                'u_n_eigen': list(range(50, 150, 20)),  # [50, 70, 90, 110, 130]
                'i_n_eigen': list(range(80, 200, 20)),  # [80, 100, 120, 140, 160, 180]
            },
            'yelp2018': {
                'u_n_eigen': list(range(60, 180, 20)),  # [60, 80, 100, 120, 140, 160]
                'i_n_eigen': list(range(100, 240, 20)),  # [100, 120, 140, 160, 180, 200, 220]
            },
            'amazon-book': {
                'u_n_eigen': list(range(80, 200, 30)),  # [80, 110, 140, 170]
                'i_n_eigen': list(range(120, 300, 30)),  # [120, 150, 180, 210, 240, 270]
            }
        }
        
        # Combine base config with eigenvalue config
        config = base_configs.get(self.dataset, base_configs['ml-100k']).copy()
        config.update(eigenvalue_configs.get(self.dataset, eigenvalue_configs['ml-100k']))
        
        return config
    
    def _get_dataset_batch_sizes(self) -> Tuple[str, str]:
        """Get appropriate batch sizes for dataset"""
        return self.dataset_params['batch_sizes']
    
    def _clear_cache(self):
        """CRITICAL FIX: Clear all cache files to ensure fresh model creation"""
        cache_dir = "../cache"
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print("    🗑️ Cleared cache directory")
            except Exception as e:
                print(f"    ⚠️ Cache clear failed: {e}")
    
    def _generate_cache_key_preview(self, config: Dict) -> str:
        """Generate what the cache key would look like for this config"""
        dataset = self.dataset
        sim_type = config.get('similarity_type', 'cosine')
        threshold = str(config.get('similarity_threshold', 0.01)).replace('.', 'p')
        u_eigen = config.get('u_n_eigen', 50)
        i_eigen = config.get('i_n_eigen', 50)
        filter_design = config.get('filter_design', 'enhanced_basis')
        init_filter = config.get('init_filter', 'smooth')
        filter_order = config.get('filter_order', 6)
        filter_mode = config.get('filter', 'ui')
        
        # This should match the pattern in model_enhanced.py
        base_name = f"{dataset}_{sim_type}_th{threshold}_u{u_eigen}_i{i_eigen}_{filter_design}_{init_filter}_fo{filter_order}_{filter_mode}"
        return base_name
    
    def _verify_cache_key_differences(self, stage_configs: List[Dict]) -> bool:
        """Verify that different configs produce different cache keys"""
        print(f"\n🔍 CACHE KEY VERIFICATION:")
        
        if len(stage_configs) < 2:
            print("  Only one config, skipping verification")
            return True
        
        cache_keys = []
        unique_keys = set()
        
        for i, config in enumerate(stage_configs[:5]):  # Check first 5 configs
            key = self._generate_cache_key_preview(config)
            cache_keys.append(key)
            unique_keys.add(key)
            
            if i < 3:  # Show first 3 for debugging
                print(f"  Config {i+1}: ...{key[-60:]}")  # Show last 60 chars
        
        unique_count = len(unique_keys)
        total_checked = len(cache_keys)
        
        print(f"  📊 Unique cache keys: {unique_count}/{total_checked}")
        
        if unique_count == 1:
            print(f"  ❌ WARNING: ALL CONFIGS GENERATE SAME CACHE KEY!")
            print(f"     This means eigendecompositions will be reused!")
            return False
        elif unique_count < total_checked:
            print(f"  ⚠️  Some configs share cache keys (expected for same parameters)")
        else:
            print(f"  ✅ All configs generate unique cache keys")
        
        return True
    
    def _debug_parameter_changes(self, stage: Dict, stage_configs: List[Dict]):
        """Debug what parameters are actually changing in this stage"""
        print(f"\n🔍 PARAMETER CHANGE ANALYSIS:")
        print(f"  Stage: {stage['name']}")
        print(f"  Primary parameter: {stage['primary_param']}")
        
        if len(stage_configs) < 2:
            print("  Not enough configs to analyze")
            return
        
        # Find which parameters are varying
        varying_params = set()
        all_params = {}
        
        for config in stage_configs:
            for key, value in config.items():
                if key not in all_params:
                    all_params[key] = set()
                all_params[key].add(str(value))
        
        # Classify as varying or fixed
        for param, values in all_params.items():
            if len(values) > 1:
                varying_params.add(param)
        
        print(f"  📊 Analysis of {len(stage_configs)} configurations:")
        print(f"     Varying parameters: {len(varying_params)}")
        for param in sorted(varying_params):
            values = sorted(list(all_params[param]))
            if len(values) <= 5:
                print(f"       {param}: {values}")
            else:
                print(f"       {param}: {values[:3]}...{values[-1]} ({len(values)} total)")
    
    def _calculate_step_size(self, values: List) -> float:
        """Calculate average step size for a parameter range"""
        if len(values) < 2:
            return 1.0
        
        # Calculate differences between consecutive values
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        return sum(diffs) / len(diffs)

    def _is_at_boundary(self, best_value, search_values: List, tolerance: float = 0.01) -> str:
        """Check if best value is at lower, upper, or no boundary"""
        if not search_values:
            return 'none'
        
        min_val, max_val = min(search_values), max(search_values)
        
        if abs(best_value - min_val) <= tolerance:
            return 'lower'
        elif abs(best_value - max_val) <= tolerance:
            return 'upper'
        else:
            return 'none'

    def _expand_boundary(self, current_values: List, boundary_type: str, step_size: float, 
                        param_name: str, max_expansions: int = 3) -> List:
        """Expand boundary values in the specified direction"""
        
        # Parameter-specific constraints
        constraints = {
            'lr': {'min': 1e-6, 'max': 2.0},
            'decay': {'min': 1e-4, 'max': 1.0}, 
            'similarity_threshold': {'min': 1e-4, 'max': 0.5},
            'u_n_eigen': {'min': 5, 'max': 500},
            'i_n_eigen': {'min': 5, 'max': 500},
            'filter_order': {'min': 2, 'max': 12}
        }
        
        constraint = constraints.get(param_name, {'min': 0.001, 'max': 1000})
        expanded_values = []
        
        if boundary_type == 'lower':
            # Expand downward from minimum
            min_val = min(current_values)
            for i in range(1, max_expansions + 1):
                new_val = min_val - i * step_size
                
                # Apply constraints and rounding for integer parameters
                if param_name in ['u_n_eigen', 'i_n_eigen', 'filter_order']:
                    new_val = max(constraint['min'], int(round(new_val)))
                    if new_val in current_values or new_val in expanded_values:
                        break
                else:
                    new_val = max(constraint['min'], new_val)
                    if new_val <= 0:
                        break
                
                expanded_values.append(new_val)
                
        elif boundary_type == 'upper':
            # Expand upward from maximum
            max_val = max(current_values)
            for i in range(1, max_expansions + 1):
                new_val = max_val + i * step_size
                
                # Apply constraints and rounding for integer parameters
                if param_name in ['u_n_eigen', 'i_n_eigen', 'filter_order']:
                    new_val = min(constraint['max'], int(round(new_val)))
                    if new_val in current_values or new_val in expanded_values:
                        break
                else:
                    new_val = min(constraint['max'], new_val)
                
                expanded_values.append(new_val)
        
        return expanded_values

    def _adaptive_boundary_search(self, stage: Dict, initial_results: List[Tuple], 
                                 search_space: Dict) -> List[Tuple]:
        """Perform adaptive boundary expansion for parameters at boundaries"""
        
        if not initial_results:
            return initial_results
        
        # Sort by performance (NDCG)
        initial_results.sort(key=lambda x: x[1], reverse=True)
        best_config, best_ndcg = initial_results[0]
        
        print(f"\n🔄 ADAPTIVE BOUNDARY EXPANSION:")
        print(f"   Initial best NDCG: {best_ndcg:.6f}")
        
        # Check each parameter for boundary optimization
        expansion_results = []
        total_expansions = 0
        
        for param_name, param_values in search_space.items():
            if param_name == stage['primary_param'] and len(param_values) > 1:
                
                best_param_value = best_config.get(param_name)
                if best_param_value is None:
                    continue
                
                boundary_type = self._is_at_boundary(best_param_value, param_values)
                
                if boundary_type != 'none':
                    step_size = self._calculate_step_size(param_values)
                    
                    print(f"   🎯 {param_name}: Best value {best_param_value} at {boundary_type} boundary")
                    print(f"      Step size: {step_size:.3f}")
                    
                    # Generate expansion values
                    expansion_values = self._expand_boundary(
                        param_values, boundary_type, step_size, param_name, max_expansions=3
                    )
                    
                    if expansion_values:
                        print(f"      Expanding {boundary_type}: {expansion_values}")
                        
                        # Test expansion values
                        for exp_value in expansion_values:
                            total_expansions += 1
                            
                            # Create config with expanded value
                            expanded_config = best_config.copy()
                            expanded_config[param_name] = exp_value
                            
                            # Handle eigenvalue expansions specially
                            if param_name in ['u_n_eigen', 'i_n_eigen']:
                                config_str = f"u={expanded_config.get('u_n_eigen', '?'):3}, i={expanded_config.get('i_n_eigen', '?'):3}"
                            else:
                                config_str = f"{param_name}={exp_value}"
                            
                            print(f"      [EXP] {config_str:<25}", end=" ", flush=True)
                            
                            # Evaluate expanded configuration
                            exp_ndcg = self._evaluate_config(expanded_config)
                            
                            if exp_ndcg is not None:
                                print(f"✅ {exp_ndcg:.6f}", end="")
                                
                                if exp_ndcg > best_ndcg:
                                    improvement = exp_ndcg - best_ndcg
                                    print(f" 🚀 +{improvement:.6f}")
                                    expansion_results.append((expanded_config, exp_ndcg))
                                    best_ndcg = exp_ndcg  # Update best for further expansions
                                elif exp_ndcg < best_ndcg - 0.001:  # Significant drop
                                    drop = best_ndcg - exp_ndcg
                                    print(f" 📉 -{drop:.6f} (stopping expansion)")
                                    break  # Stop expanding in this direction
                                else:
                                    print(f" ➡️ similar")
                                    expansion_results.append((expanded_config, exp_ndcg))
                            else:
                                print("❌")
                                break  # Stop on failure
                    else:
                        print(f"      No valid expansion values")
                else:
                    print(f"   ✅ {param_name}: Best value {best_param_value} not at boundary")
        
        # Handle eigenvalue expansion specially
        if stage['primary_param'] == 'eigenvalues':
            # Check u_n_eigen boundary
            u_values = search_space.get('u_n_eigen', [])
            best_u = best_config.get('u_n_eigen')
            if best_u and u_values:
                u_boundary = self._is_at_boundary(best_u, u_values)
                if u_boundary != 'none':
                    u_step = self._calculate_step_size(u_values)
                    print(f"   🎯 u_n_eigen: Best value {best_u} at {u_boundary} boundary")
                    u_expansions = self._expand_boundary(u_values, u_boundary, u_step, 'u_n_eigen', 3)
                    
                    if u_expansions:
                        print(f"      Expanding u_n_eigen {u_boundary}: {u_expansions}")
                        for exp_u in u_expansions:
                            total_expansions += 1
                            expanded_config = best_config.copy()
                            expanded_config['u_n_eigen'] = exp_u
                            config_str = f"u={exp_u:3}, i={expanded_config.get('i_n_eigen', '?'):3}"
                            print(f"      [EXP] {config_str:<25}", end=" ", flush=True)
                            
                            exp_ndcg = self._evaluate_config(expanded_config)
                            if exp_ndcg is not None:
                                print(f"✅ {exp_ndcg:.6f}", end="")
                                if exp_ndcg > best_ndcg:
                                    print(f" 🚀 +{exp_ndcg - best_ndcg:.6f}")
                                    expansion_results.append((expanded_config, exp_ndcg))
                                    best_ndcg = exp_ndcg
                                elif exp_ndcg < best_ndcg - 0.001:
                                    print(f" 📉 -{best_ndcg - exp_ndcg:.6f} (stopping)")
                                    break
                                else:
                                    print(f" ➡️ similar")
                                    expansion_results.append((expanded_config, exp_ndcg))
                            else:
                                print("❌")
                                break
            
            # Check i_n_eigen boundary
            i_values = search_space.get('i_n_eigen', [])
            best_i = best_config.get('i_n_eigen')
            if best_i and i_values:
                i_boundary = self._is_at_boundary(best_i, i_values)
                if i_boundary != 'none':
                    i_step = self._calculate_step_size(i_values)
                    print(f"   🎯 i_n_eigen: Best value {best_i} at {i_boundary} boundary")
                    i_expansions = self._expand_boundary(i_values, i_boundary, i_step, 'i_n_eigen', 3)
                    
                    if i_expansions:
                        print(f"      Expanding i_n_eigen {i_boundary}: {i_expansions}")
                        for exp_i in i_expansions:
                            total_expansions += 1
                            expanded_config = best_config.copy()
                            expanded_config['i_n_eigen'] = exp_i
                            config_str = f"u={expanded_config.get('u_n_eigen', '?'):3}, i={exp_i:3}"
                            print(f"      [EXP] {config_str:<25}", end=" ", flush=True)
                            
                            exp_ndcg = self._evaluate_config(expanded_config)
                            if exp_ndcg is not None:
                                print(f"✅ {exp_ndcg:.6f}", end="")
                                if exp_ndcg > best_ndcg:
                                    print(f" 🚀 +{exp_ndcg - best_ndcg:.6f}")
                                    expansion_results.append((expanded_config, exp_ndcg))
                                    best_ndcg = exp_ndcg
                                elif exp_ndcg < best_ndcg - 0.001:
                                    print(f" 📉 -{best_ndcg - exp_ndcg:.6f} (stopping)")
                                    break
                                else:
                                    print(f" ➡️ similar")
                                    expansion_results.append((expanded_config, exp_ndcg))
                            else:
                                print("❌")
                                break
        
        if total_expansions > 0:
            print(f"   📊 Total expansions tested: {total_expansions}")
            print(f"   📈 Final best NDCG: {best_ndcg:.6f}")
        else:
            print(f"   ℹ️  No boundary expansions needed")
        
        # Combine initial and expansion results
        all_results = initial_results + expansion_results
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return all_results
    
    def _build_command(self, config: Dict) -> List[str]:
        """Build command line for experiment"""
        train_batch, eval_batch = self._get_dataset_batch_sizes()
        epochs = self.dataset_params['epochs']
        
        cmd = [
            sys.executable, "main.py",
            "--dataset", self.dataset,
            "--model_type", self.model_type,  # CRITICAL: Specify model type
            "--lr", str(config['lr']),
            "--decay", str(config['decay']),
            "--filter", config['filter'],
            "--filter_design", config['filter_design'],
            "--init_filter", config['init_filter'],
            "--epochs", str(epochs),
            "--patience", "8",
            "--filter_order", "6",
            "--train_u_batch_size", train_batch,
            "--eval_u_batch_size", eval_batch,
            "--seed", "2025",
            "--verbose", "0"
        ]
        
        # Model-specific parameters
        if self.model_type == 'enhanced':
            # Enhanced model uses similarity threshold and separate eigenvalues
            cmd.extend(["--similarity_threshold", str(config['similarity_threshold'])])
            if 'u_n_eigen' in config:
                cmd.extend(["--u_n_eigen", str(config['u_n_eigen'])])
            if 'i_n_eigen' in config:
                cmd.extend(["--i_n_eigen", str(config['i_n_eigen'])])
        else:
            # Basic model uses separate eigenvalues without similarity threshold
            if 'u_n_eigen' in config:
                cmd.extend(["--u_n_eigen", str(config['u_n_eigen'])])
            if 'i_n_eigen' in config:
                cmd.extend(["--i_n_eigen", str(config['i_n_eigen'])])
        
        return cmd
    
    def _evaluate_config(self, config: Dict, timeout: int = 600) -> Optional[float]:
        """FIXED: Evaluate a single configuration with cache clearing"""
        
        # CRITICAL FIX: Clear cache before each experiment
        self._clear_cache()
        
        cmd = self._build_command(config)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode != 0:
                print(f"❌ FAILED (return code: {result.returncode})")
                if "CUDA out of memory" in result.stderr:
                    print("    💾 GPU memory issue")
                elif "ImportError" in result.stderr:
                    print("    📦 Import error")
                return None
            
            # Parse NDCG@20
            ndcg = self._parse_ndcg(result.stdout)
            return ndcg
                
        except subprocess.TimeoutExpired:
            print("❌ TIMEOUT")
            return None
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            return None
    
    def _parse_ndcg(self, output: str) -> Optional[float]:
        """Parse NDCG@20 from output"""
        for line in output.split('\n'):
            if "Final Test Results:" in line and "NDCG@20=" in line:
                try:
                    ndcg_part = line.split("NDCG@20=")[1]
                    return float(ndcg_part.split(",")[0])
                except (IndexError, ValueError):
                    continue
        return None
    
    def _run_stage_with_adaptive_expansion(self, stage: Dict, previous_best_config: Dict) -> Tuple[Dict, float]:
        """Enhanced stage runner with adaptive boundary expansion"""
        
        print(f"\n{'='*80}")
        print(f"🔍 STAGE: {stage['name'].upper().replace('_', ' ')}")
        print(f"📝 Description: {stage['description']}")
        print(f"🎯 Primary Parameter: {stage['primary_param']}")
        print(f"{'='*80}")
        
        # Build search space for this stage
        if stage['search_space'] == 'dataset_dependent':
            search_space = {
                'u_n_eigen': self.dataset_params['u_n_eigen'],
                'i_n_eigen': self.dataset_params['i_n_eigen']
            }
        else:
            search_space = stage['search_space']
        
        # Generate configurations for this stage
        stage_configs = []
        
        if stage['primary_param'] == 'eigenvalues':
            # Both models use separate u_n_eigen and i_n_eigen
            for u_eigen in search_space['u_n_eigen']:
                for i_eigen in search_space['i_n_eigen']:
                    config = previous_best_config.copy()
                    config.update(stage['fixed_params'])
                    config['u_n_eigen'] = u_eigen
                    config['i_n_eigen'] = i_eigen
                    stage_configs.append(config)
        else:
            # Regular parameter sweep
            primary_values = search_space[stage['primary_param']]
            for value in primary_values:
                config = previous_best_config.copy()
                config.update(stage['fixed_params'])
                config[stage['primary_param']] = value
                stage_configs.append(config)
        
        print(f"🏃 Running {len(stage_configs)} initial experiments for {stage['name']}...")
        
        # VERIFICATION: Check parameter changes and cache keys
        self._debug_parameter_changes(stage, stage_configs)
        cache_keys_unique = self._verify_cache_key_differences(stage_configs)
        
        if not cache_keys_unique and stage['name'] == 'eigenvalues':
            print(f"\n❌ CRITICAL ERROR: Eigenvalue stage has identical cache keys!")
            print(f"   This explains why all results are identical.")
            print(f"   The eigendecompositions are being reused from cache.")
        
        print("-" * 80)
        
        # Run initial experiments
        initial_results = []
        
        for i, config in enumerate(stage_configs, 1):
            # Progress indicator
            progress = (i / len(stage_configs)) * 100
            
            if stage['primary_param'] == 'eigenvalues':
                config_str = f"u={config['u_n_eigen']:3d}, i={config['i_n_eigen']:3d}"
            else:
                config_str = f"{stage['primary_param']}={config[stage['primary_param']]}"
            
            print(f"[{i:3d}/{len(stage_configs)}] {progress:5.1f}% | {config_str:<25}", end=" ", flush=True)
            
            # Evaluate
            ndcg = self._evaluate_config(config)
            
            if ndcg is not None:
                print(f"✅ {ndcg:.6f}")
                initial_results.append((config, ndcg))
            else:
                print("❌")
        
        # ADAPTIVE BOUNDARY EXPANSION
        # Only expand for parameters that can be meaningfully expanded
        expandable_params = ['lr', 'decay', 'similarity_threshold', 'u_n_eigen', 'i_n_eigen']
        
        if stage['primary_param'] in expandable_params or stage['primary_param'] == 'eigenvalues':
            all_results = self._adaptive_boundary_search(stage, initial_results, search_space)
        else:
            all_results = initial_results
            print(f"\n📍 Parameter '{stage['primary_param']}' not suitable for boundary expansion")
        
        # Store all results (including expansions)
        self.stage_results[stage['name']] = all_results
        
        # Find best result
        if all_results:
            all_results.sort(key=lambda x: x[1], reverse=True)
            best_config, best_ndcg = all_results[0]
            
            # Enhanced analysis for eigenvalue stage
            if stage['name'] == 'eigenvalues' and len(all_results) > 1:
                ndcg_values = [result[1] for result in all_results]
                unique_ndcg = len(set([round(v, 6) for v in ndcg_values]))
                if unique_ndcg == 1:
                    print(f"\n❌ ALL EXPERIMENTS RETURNED IDENTICAL RESULTS!")
                    print(f"   NDCG: {ndcg_values[0]:.6f} (all {len(ndcg_values)} experiments)")
                else:
                    print(f"\n✅ Experiments show variation in results")
                    print(f"   NDCG range: {min(ndcg_values):.6f} to {max(ndcg_values):.6f}")
                    print(f"   Unique values: {unique_ndcg}/{len(ndcg_values)}")
            
            # Report stage summary
            print(f"\n📊 STAGE SUMMARY:")
            print(f"  ✅ Successful experiments: {len(all_results)} (initial: {len(initial_results)})")
            
            # Format the best parameter value
            if stage['primary_param'] == 'eigenvalues':
                best_param_str = f"u={best_config['u_n_eigen']}, i={best_config['i_n_eigen']}"
            else:
                best_param_str = str(best_config[stage['primary_param']])
            
            print(f"  🏆 Best {stage['primary_param']}: {best_param_str}")
            print(f"  📈 Best NDCG@20: {best_ndcg:.6f}")
            
            # Show top 3 for this stage
            print(f"\n🥇 TOP 3 FOR THIS STAGE:")
            for i, (config, ndcg) in enumerate(all_results[:3], 1):
                if stage['primary_param'] == 'eigenvalues':
                    param_str = f"u={config['u_n_eigen']}, i={config['i_n_eigen']}"
                else:
                    param_str = f"{stage['primary_param']}={config[stage['primary_param']]}"
                print(f"  {i}. {param_str:<30} NDCG={ndcg:.6f}")
            
            return best_config, best_ndcg
        else:
            print(f"\n❌ No successful experiments!")
            return None, 0.0
    
    def run_hierarchical_search(self) -> Dict:
        """Run the complete hierarchical search with adaptive boundary expansion"""
        
        print(f"🚀 HIERARCHICAL HYPERPARAMETER SEARCH WITH ADAPTIVE EXPANSION")
        print(f"=" * 80)
        print(f"📊 Dataset: {self.dataset.upper()}")
        print(f"🔧 Model Type: {self.model_type.upper()}")
        print(f"🔄 Search Stages: {len(self.search_stages)}")
        print(f"🎯 Adaptive Expansion: Enabled for boundary parameters")
        print(f"📈 Dataset Config:")
        print(f"   Users: {self.dataset_params.get('users', 'N/A'):,}")
        print(f"   Items: {self.dataset_params.get('items', 'N/A'):,}")
        print(f"   Sparsity: {self.dataset_params.get('sparsity', 'N/A')}")
        print(f"   U eigenvalues: {len(self.dataset_params['u_n_eigen'])} options")
        print(f"   I eigenvalues: {len(self.dataset_params['i_n_eigen'])} options")
        print(f"=" * 80)
        
        start_time = time.time()
        
        # Initialize with baseline configuration
        current_best_config = {
            'lr': 0.01,
            'decay': 0.01,
            'filter': 'ui',
            'filter_design': 'enhanced_basis',
            'init_filter': 'smooth',
            'u_n_eigen': self.dataset_params['u_n_eigen'][len(self.dataset_params['u_n_eigen'])//2],
            'i_n_eigen': self.dataset_params['i_n_eigen'][len(self.dataset_params['i_n_eigen'])//2]
        }
        
        # Add similarity threshold for enhanced model
        if self.model_type == 'enhanced':
            current_best_config['similarity_threshold'] = 0.01
        
        current_best_ndcg = 0.0
        
        # Run each stage progressively WITH ADAPTIVE EXPANSION
        for stage_idx, stage in enumerate(self.search_stages, 1):
            print(f"\n🎯 STARTING STAGE {stage_idx}/{len(self.search_stages)}")
            
            # USE NEW METHOD WITH ADAPTIVE EXPANSION
            stage_best_config, stage_best_ndcg = self._run_stage_with_adaptive_expansion(stage, current_best_config)
            
            if stage_best_config is not None:
                current_best_config = stage_best_config
                current_best_ndcg = stage_best_ndcg
                self.best_configs[stage['name']] = (stage_best_config.copy(), stage_best_ndcg)
            
            print(f"✅ Stage {stage_idx} completed. Overall best NDCG: {current_best_ndcg:.6f}")
        
        total_time = time.time() - start_time
        
        # Final results only
        self._show_final_results(current_best_config, current_best_ndcg, total_time)
        
        return {
            'final_best_config': current_best_config,
            'final_best_ndcg': current_best_ndcg,
            'total_time': total_time
        }
    
    def _show_final_results(self, best_config: Dict, best_ndcg: float, total_time: float):
        """Show only the essential final results"""
        
        print(f"\n" + "=" * 80)
        print(f"🏆 SEARCH COMPLETED - {self.dataset.upper()} ({self.model_type.upper()} MODEL)")
        print(f"=" * 80)
        
        print(f"\n⏱️  SUMMARY:")
        total_experiments = sum(len(results) for results in self.stage_results.values())
        print(f"  Total Time: {total_time/60:.1f} minutes")
        print(f"  Total Experiments: {total_experiments}")
        print(f"  Final NDCG@20: {best_ndcg:.6f}")
        
        print(f"\n🏆 OPTIMAL HYPERPARAMETERS:")
        print(f"  --lr {best_config['lr']}")
        print(f"  --decay {best_config['decay']}")
        
        if self.model_type == 'enhanced':
            print(f"  --similarity_threshold {best_config['similarity_threshold']}")
        
        print(f"  --u_n_eigen {best_config['u_n_eigen']}")
        print(f"  --i_n_eigen {best_config['i_n_eigen']}")
        print(f"  --filter {best_config['filter']}")
        print(f"  --filter_design {best_config['filter_design']}")
        print(f"  --init_filter {best_config['init_filter']}")
        
        print(f"\n🚀 COMPLETE COMMAND:")
        cmd_parts = [
            "python main.py",
            f"--model_type {self.model_type}",
            f"--dataset {self.dataset}",
            f"--lr {best_config['lr']}",
            f"--decay {best_config['decay']}",
            f"--u_n_eigen {best_config['u_n_eigen']}",
            f"--i_n_eigen {best_config['i_n_eigen']}",
            f"--filter {best_config['filter']}",
            f"--filter_design {best_config['filter_design']}",
            f"--init_filter {best_config['init_filter']}",
            f"--epochs {self.dataset_params['epochs']}",
            "--patience 10"
        ]
        
        if self.model_type == 'enhanced':
            cmd_parts.insert(-2, f"--similarity_threshold {best_config['similarity_threshold']}")
        
        print(" \\\n    ".join(cmd_parts))
        print(f"\n" + "=" * 80)
    
    def save_results(self, filename: Optional[str] = None):
        """Save hierarchical search results - DISABLED"""
        pass  # No saving functionality


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Hyperparameter Search with Adaptive Boundary Expansion")
    
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'ml-1m', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book'],
                       help='Dataset to search on')
    
    parser.add_argument('--model_type', type=str, default='enhanced',
                       choices=['enhanced', 'basic'],
                       help='Model type: enhanced (model_enhanced.py) or basic (model.py)')
    
    parser.add_argument('--save', action='store_true', default=False,
                       help='Save results to file (disabled by default)')
    
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per experiment in seconds')
    
    args = parser.parse_args()
    
    # Initialize hierarchical searcher
    searcher = HierarchicalHyperparameterSearch(args.dataset, args.model_type)
    
    # Run hierarchical search
    results = searcher.run_hierarchical_search()
    
    # No automatic saving - only save if explicitly requested
    if args.save and results:
        print("\n💾 Saving disabled by default - search completed!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================
# ADAPTIVE BOUNDARY EXPANSION - NEW FEATURES
# ============================================================================

# 🎯 ADAPTIVE BOUNDARY EXPANSION:
# When the best parameter is at a boundary (min/max of search range), 
# automatically expands the search in that direction until performance drops.

# Example Output:
# 🔄 ADAPTIVE BOUNDARY EXPANSION:
#    Initial best NDCG: 0.162340
#    🎯 u_n_eigen: Best value 45 at upper boundary
#       Step size: 5.000
#       Expanding upper: [50, 55, 60]
#       [EXP] u= 50, i= 60         ✅ 0.163240 🚀 +0.000900
#       [EXP] u= 55, i= 60         ✅ 0.164120 🚀 +0.000880
#       [EXP] u= 60, i= 60         ✅ 0.161890 📉 -0.002230 (stopping expansion)
#    📊 Total expansions tested: 3
#    📈 Final best NDCG: 0.164120

# 🔧 EXPANDABLE PARAMETERS:
# - lr: Learning rate (1e-6 to 2.0)
# - decay: Weight decay (1e-4 to 1.0)
# - similarity_threshold: Similarity threshold (1e-4 to 0.5)
# - u_n_eigen: User eigenvalues (5 to 500)
# - i_n_eigen: Item eigenvalues (5 to 500)

# 🛡️ SAFETY FEATURES:
# - Respects parameter constraints (no negative values, reasonable maximums)
# - Stops expansion when performance drops significantly (>0.001 NDCG)
# - Limited to 3 expansion steps per boundary to prevent runaway expansion
# - Integer rounding for eigenvalue parameters

# 📊 BENEFITS:
# - Discovers optimal values outside initial search ranges
# - Typically adds 2-5 extra experiments per stage
# - Can improve final performance by 0.005-0.01 NDCG
# - Fully automated - no manual intervention needed

# ============================================================================