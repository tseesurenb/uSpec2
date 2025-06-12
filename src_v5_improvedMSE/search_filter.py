'''
Created on June 12, 2025
Enhanced Comprehensive Filter Search for Universal Spectral CF
Search all filter combinations with better coverage and test data evaluation

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import os
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import subprocess
import argparse
from collections import defaultdict
import random

# Available filter types and initialization patterns
FILTER_TYPES = [
    'original',              # Basic universal filter
    'spectral_basis',        # Spectral basis filter
    'enhanced_basis',        # Enhanced multi-pattern filter
    'chebyshev',            # Chebyshev polynomial filter
    'jacobi',               # Jacobi polynomial filter
    'legendre',             # Legendre polynomial filter
    'laguerre',             # Laguerre polynomial filter
    'hermite',              # Hermite polynomial filter
    'bernstein',            # Bernstein polynomial filter
    'bandstop',             # Band-stop filter
    'adaptive_bandstop',    # Advanced multi-band stop filter
    'parametric',           # Parametric multi-band filter
    'multiscale',           # Multi-scale spectral filter
    'harmonic',             # Harmonic series filter
    'golden',               # Adaptive golden ratio filter
    'ensemble'              # Ensemble of all filters
]

INIT_PATTERNS = [
    'smooth', 'sharp', 'bandpass', 'golden_036', 
    'butterworth', 'gaussian', 'band_stop', 'notch'
]

# Known good configurations to prioritize
PRIORITY_CONFIGS = [
    # Your manually found good configuration
    ('multiscale', 'smooth', 'chebyshev', 'sharp', 'original', 'smooth'),
    # Other promising combinations
    ('chebyshev', 'sharp', 'parametric', 'butterworth', 'laguerre', 'gaussian'),
    ('enhanced_basis', 'smooth', 'chebyshev', 'sharp', 'original', 'smooth'),
    ('spectral_basis', 'golden_036', 'multiscale', 'butterworth', 'legendre', 'smooth'),
    ('original', 'smooth', 'original', 'sharp', 'original', 'smooth'),
]

class EnhancedFilterSearcher:
    def __init__(self, dataset='ml-100k', base_args=None, output_dir='search_results'):
        self.dataset = dataset
        self.base_args = base_args or {}
        self.output_dir = output_dir
        self.results = []
        self.tested_configs = set()  # Track tested configurations to avoid duplicates
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(output_dir, f'enhanced_filter_search_{dataset}_{timestamp}.json')
        self.csv_file = os.path.join(output_dir, f'enhanced_filter_search_{dataset}_{timestamp}.csv')
        
        print(f"🔍 Enhanced Filter Search initialized for {dataset}")
        print(f"📊 Results will be saved to: {self.results_file}")
        print(f"📈 CSV summary will be saved to: {self.csv_file}")
    
    def config_to_key(self, config):
        """Convert config to a hashable key for deduplication"""
        return (
            config.get('user_filter_design'),
            config.get('user_init_filter'),
            config.get('item_filter_design'),
            config.get('item_init_filter'),
            config.get('bipartite_filter_design'),
            config.get('bipartite_init_filter')
        )
    
    def run_single_experiment(self, config, use_test=False):
        """Run a single experiment with given configuration"""
        
        # Check if already tested
        config_key = self.config_to_key(config)
        if config_key in self.tested_configs:
            print(f"⏭️ Skipping duplicate configuration")
            return None
        
        self.tested_configs.add(config_key)
        
        # Build command
        cmd = ['python', 'main.py']
        cmd.extend(['--dataset', self.dataset])
        
        # Add base arguments
        for key, value in self.base_args.items():
            if key not in ['dataset']:  # Skip dataset as it's already added
                cmd.extend([f'--{key}', str(value)])
        
        # Add filter configuration
        for key, value in config.items():
            cmd.extend([f'--{key}', str(value)])
        
        # Add test flag if needed
        if use_test:
            cmd.extend(['--use_test_for_validation'])
        
        # Run experiment
        try:
            config_str = f"U:{config.get('user_filter_design')}({config.get('user_init_filter')}) | " \
                        f"I:{config.get('item_filter_design')}({config.get('item_init_filter')}) | " \
                        f"B:{config.get('bipartite_filter_design')}({config.get('bipartite_init_filter')})"
            print(f"🚀 Running: {config_str}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse results from output
                output = result.stdout
                metrics = self.parse_metrics(output)
                
                if metrics:
                    experiment_result = {
                        'config': config,
                        'metrics': metrics,
                        'command': ' '.join(cmd),
                        'timestamp': datetime.now().isoformat(),
                        'status': 'success',
                        'use_test': use_test
                    }
                    self.results.append(experiment_result)
                    
                    test_str = "(TEST)" if use_test else "(VAL)"
                    print(f"✅ Success {test_str}: NDCG@20={metrics.get('ndcg20', 'N/A'):.4f}")
                    return metrics
                else:
                    print(f"❌ Failed to parse metrics")
                    experiment_result = {
                        'config': config,
                        'command': ' '.join(cmd),
                        'timestamp': datetime.now().isoformat(),
                        'status': 'parse_failed',
                        'stdout': output[-500:],
                        'stderr': result.stderr[-500:] if result.stderr else '',
                        'use_test': use_test
                    }
                    self.results.append(experiment_result)
                    return None
            else:
                print(f"❌ Command failed (code {result.returncode})")
                if result.stderr:
                    print(f"   Error: {result.stderr[-200:]}")
                
                experiment_result = {
                    'config': config,
                    'command': ' '.join(cmd),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'command_failed',
                    'returncode': result.returncode,
                    'stderr': result.stderr[-500:] if result.stderr else '',
                    'use_test': use_test
                }
                self.results.append(experiment_result)
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout after 300s")
            experiment_result = {
                'config': config,
                'command': ' '.join(cmd),
                'timestamp': datetime.now().isoformat(),
                'status': 'timeout',
                'use_test': use_test
            }
            self.results.append(experiment_result)
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            experiment_result = {
                'config': config,
                'command': ' '.join(cmd),
                'timestamp': datetime.now().isoformat(),
                'status': 'exception',
                'error': str(e),
                'use_test': use_test
            }
            self.results.append(experiment_result)
            return None
    
    def parse_metrics(self, output):
        """Parse metrics from command output, handling ANSI color codes"""
        try:
            import re
            
            # Remove ANSI color codes first
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            clean_output = ansi_escape.sub('', output)
            
            lines = clean_output.split('\n')
            for line in lines:
                if ('R@20:' in line and 'P@20:' in line and 'NDCG@20:' in line) or \
                   ('Recall@20:' in line and 'Precision@20:' in line and 'NDCG@20:' in line):
                    
                    if '|' in line:
                        # Format: "📊 R@20: 0.3098 | P@20: 0.2528 | NDCG@20: 0.3630"
                        parts = line.split('|')
                        recall_part = parts[0].split(':')[1].strip()
                        precision_part = parts[1].split(':')[1].strip()
                        ndcg_part = parts[2].split(':')[1].strip()
                        
                        recall = float(recall_part)
                        precision = float(precision_part)
                        ndcg = float(ndcg_part)
                    else:
                        # Format: "Recall@20: 0.307369, Precision@20: 0.252386, NDCG@20: 0.361948"
                        recall_match = re.search(r'(?:Recall@20|R@20):\s*([0-9.]+)', line)
                        precision_match = re.search(r'(?:Precision@20|P@20):\s*([0-9.]+)', line)
                        ndcg_match = re.search(r'NDCG@20:\s*([0-9.]+)', line)
                        
                        if recall_match and precision_match and ndcg_match:
                            recall = float(recall_match.group(1))
                            precision = float(precision_match.group(1))
                            ndcg = float(ndcg_match.group(1))
                        else:
                            continue
                    
                    return {
                        'recall20': recall,
                        'precision20': precision,
                        'ndcg20': ndcg
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Error parsing metrics: {e}")
            return None
    
    def search_priority_configs(self):
        """Test priority configurations first"""
        print(f"\n🎯 Phase 0: Testing Priority Configurations")
        print("=" * 60)
        
        priority_results = []
        
        for i, (u_filter, u_init, i_filter, i_init, b_filter, b_init) in enumerate(PRIORITY_CONFIGS, 1):
            print(f"\n[{i}/{len(PRIORITY_CONFIGS)}] Priority Config {i}")
            
            config = {
                'filter': 'uib',
                'user_filter_design': u_filter,
                'user_init_filter': u_init,
                'item_filter_design': i_filter,
                'item_init_filter': i_init,
                'bipartite_filter_design': b_filter,
                'bipartite_init_filter': b_init,
            }
            
            metrics = self.run_single_experiment(config)
            if metrics:
                priority_results.append({
                    'config': config,
                    'metrics': metrics,
                    'priority_rank': i
                })
            
            # Save intermediate results
            self.save_results()
        
        return priority_results
    
    def search_comprehensive_random(self, max_experiments=200):
        """Comprehensive random search across all combinations"""
        print(f"\n🔍 Phase 1: Comprehensive Random Search (max {max_experiments} experiments)")
        print("=" * 70)
        
        # Generate all possible combinations
        all_combinations = []
        for u_filter, u_init, i_filter, i_init, b_filter, b_init in itertools.product(
            FILTER_TYPES, INIT_PATTERNS, FILTER_TYPES, INIT_PATTERNS, FILTER_TYPES, INIT_PATTERNS
        ):
            config = {
                'filter': 'uib',
                'user_filter_design': u_filter,
                'user_init_filter': u_init,
                'item_filter_design': i_filter,
                'item_init_filter': i_init,
                'bipartite_filter_design': b_filter,
                'bipartite_init_filter': b_init,
            }
            
            # Skip if already tested
            if self.config_to_key(config) not in self.tested_configs:
                all_combinations.append(config)
        
        print(f"🎲 Total possible combinations: {len(FILTER_TYPES)**3 * len(INIT_PATTERNS)**3:,}")
        print(f"🔢 Remaining untested combinations: {len(all_combinations):,}")
        
        # Shuffle for random sampling
        random.shuffle(all_combinations)
        
        # Limit to max_experiments
        test_combinations = all_combinations[:max_experiments]
        
        successful_results = []
        
        for i, config in enumerate(test_combinations, 1):
            print(f"\n[{i}/{len(test_combinations)}] Random Search")
            
            metrics = self.run_single_experiment(config)
            if metrics:
                successful_results.append({
                    'config': config,
                    'metrics': metrics,
                    'experiment_id': i
                })
            
            # Save intermediate results every 10 experiments
            if i % 10 == 0:
                self.save_results()
                
                # Print current best
                if successful_results:
                    current_best = max(successful_results, key=lambda x: x['metrics']['ndcg20'])
                    print(f"   📈 Current best NDCG@20: {current_best['metrics']['ndcg20']:.4f}")
        
        return successful_results
    
    def search_neighborhood_optimization(self, top_configs, neighborhood_size=50):
        """Search neighborhoods around top configurations"""
        print(f"\n🎯 Phase 2: Neighborhood Optimization (top {len(top_configs)} configs)")
        print("=" * 70)
        
        neighborhood_results = []
        
        for base_idx, base_result in enumerate(top_configs, 1):
            base_config = base_result['config']
            print(f"\n[{base_idx}/{len(top_configs)}] Exploring neighborhood of:")
            config_str = f"U:{base_config['user_filter_design']}({base_config['user_init_filter']}) | " \
                        f"I:{base_config['item_filter_design']}({base_config['item_init_filter']}) | " \
                        f"B:{base_config['bipartite_filter_design']}({base_config['bipartite_init_filter']})"
            print(f"   Base: {config_str} (NDCG: {base_result['metrics']['ndcg20']:.4f})")
            
            # Generate neighborhood by varying one component at a time
            neighbors = []
            
            # Vary user filter
            for filter_type in FILTER_TYPES:
                if filter_type != base_config['user_filter_design']:
                    neighbor = base_config.copy()
                    neighbor['user_filter_design'] = filter_type
                    neighbors.append(neighbor)
            
            # Vary user init
            for init_pattern in INIT_PATTERNS:
                if init_pattern != base_config['user_init_filter']:
                    neighbor = base_config.copy()
                    neighbor['user_init_filter'] = init_pattern
                    neighbors.append(neighbor)
            
            # Vary item filter
            for filter_type in FILTER_TYPES:
                if filter_type != base_config['item_filter_design']:
                    neighbor = base_config.copy()
                    neighbor['item_filter_design'] = filter_type
                    neighbors.append(neighbor)
            
            # Vary item init
            for init_pattern in INIT_PATTERNS:
                if init_pattern != base_config['item_init_filter']:
                    neighbor = base_config.copy()
                    neighbor['item_init_filter'] = init_pattern
                    neighbors.append(neighbor)
            
            # Vary bipartite filter
            for filter_type in FILTER_TYPES:
                if filter_type != base_config['bipartite_filter_design']:
                    neighbor = base_config.copy()
                    neighbor['bipartite_filter_design'] = filter_type
                    neighbors.append(neighbor)
            
            # Vary bipartite init
            for init_pattern in INIT_PATTERNS:
                if init_pattern != base_config['bipartite_init_filter']:
                    neighbor = base_config.copy()
                    neighbor['bipartite_init_filter'] = init_pattern
                    neighbors.append(neighbor)
            
            # Remove duplicates and already tested
            unique_neighbors = []
            for neighbor in neighbors:
                if self.config_to_key(neighbor) not in self.tested_configs:
                    unique_neighbors.append(neighbor)
            
            # Limit neighborhood size
            if len(unique_neighbors) > neighborhood_size:
                unique_neighbors = random.sample(unique_neighbors, neighborhood_size)
            
            print(f"   Testing {len(unique_neighbors)} neighbors...")
            
            # Test neighbors
            for neighbor_idx, neighbor in enumerate(unique_neighbors, 1):
                print(f"   [{neighbor_idx}/{len(unique_neighbors)}] Neighbor test")
                
                metrics = self.run_single_experiment(neighbor)
                if metrics:
                    neighborhood_results.append({
                        'config': neighbor,
                        'metrics': metrics,
                        'base_config': base_config,
                        'base_ndcg': base_result['metrics']['ndcg20']
                    })
                    
                    # Check if improvement
                    if metrics['ndcg20'] > base_result['metrics']['ndcg20']:
                        improvement = metrics['ndcg20'] - base_result['metrics']['ndcg20']
                        print(f"   🎉 Improvement found! NDCG@20: {metrics['ndcg20']:.4f} (+{improvement:.4f})")
            
            # Save intermediate results
            self.save_results()
        
        return neighborhood_results
    
    def final_test_evaluation(self, top_k=10):
        """Evaluate top configurations on test data"""
        print(f"\n🏆 Phase 3: Final Test Evaluation (top {top_k} configs)")
        print("=" * 60)
        
        # Get top configurations from all successful results
        successful_results = [r for r in self.results if r['status'] == 'success' and not r.get('use_test', False)]
        
        if not successful_results:
            print("❌ No successful validation results to evaluate")
            return []
        
        # Sort by NDCG and get top k
        top_configs = sorted(successful_results, key=lambda x: x['metrics']['ndcg20'], reverse=True)[:top_k]
        
        print(f"🔄 Re-evaluating top {len(top_configs)} configurations on TEST data...")
        
        test_results = []
        
        for i, result in enumerate(top_configs, 1):
            config = result['config']
            val_ndcg = result['metrics']['ndcg20']
            
            print(f"\n[{i}/{len(top_configs)}] Test Evaluation")
            config_str = f"U:{config['user_filter_design']}({config['user_init_filter']}) | " \
                        f"I:{config['item_filter_design']}({config['item_init_filter']}) | " \
                        f"B:{config['bipartite_filter_design']}({config['bipartite_init_filter']})"
            print(f"   Config: {config_str}")
            print(f"   Validation NDCG@20: {val_ndcg:.4f}")
            
            # Run with test data
            test_metrics = self.run_single_experiment(config, use_test=True)
            
            if test_metrics:
                test_results.append({
                    'config': config,
                    'validation_metrics': result['metrics'],
                    'test_metrics': test_metrics,
                    'val_test_diff': test_metrics['ndcg20'] - val_ndcg
                })
                
                print(f"   Test NDCG@20: {test_metrics['ndcg20']:.4f} (diff: {test_metrics['ndcg20'] - val_ndcg:+.4f})")
            
            # Save intermediate results
            self.save_results()
        
        return test_results
    
    def save_results(self):
        """Save current results to files"""
        # Save JSON
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV summary
        if self.results:
            rows = []
            for result in self.results:
                if result['status'] == 'success':
                    config = result['config']
                    metrics = result['metrics']
                    
                    row = {
                        'user_filter': config.get('user_filter_design', 'N/A'),
                        'user_init': config.get('user_init_filter', 'N/A'),
                        'item_filter': config.get('item_filter_design', 'N/A'),
                        'item_init': config.get('item_init_filter', 'N/A'),
                        'bipartite_filter': config.get('bipartite_filter_design', 'N/A'),
                        'bipartite_init': config.get('bipartite_init_filter', 'N/A'),
                        'recall20': metrics['recall20'],
                        'precision20': metrics['precision20'],
                        'ndcg20': metrics['ndcg20'],
                        'use_test': result.get('use_test', False),
                        'timestamp': result['timestamp']
                    }
                    rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(self.csv_file, index=False)
    
    def run_enhanced_search(self, max_random_experiments=200, neighborhood_size=50, final_test_top_k=10):
        """Run enhanced comprehensive search pipeline"""
        start_time = time.time()
        
        print(f"🚀 Starting Enhanced Comprehensive Filter Search for {self.dataset}")
        print(f"📊 Total filter types: {len(FILTER_TYPES)}")
        print(f"🎯 Total init patterns: {len(INIT_PATTERNS)}")
        print(f"🔢 Total possible combinations: {len(FILTER_TYPES)**3 * len(INIT_PATTERNS)**3:,}")
        print(f"🎲 Random experiments: {max_random_experiments}")
        print(f"🎯 Neighborhood size: {neighborhood_size}")
        print(f"🏆 Final test top-k: {final_test_top_k}")
        
        # Phase 0: Test priority configurations
        priority_results = self.search_priority_configs()
        
        # Phase 1: Comprehensive random search
        random_results = self.search_comprehensive_random(max_random_experiments)
        
        # Get top configurations for neighborhood search
        all_successful = priority_results + random_results
        if all_successful:
            # Sort by NDCG and get top configurations for neighborhood search
            top_for_neighborhood = sorted(all_successful, key=lambda x: x['metrics']['ndcg20'], reverse=True)[:5]
            
            # Phase 2: Neighborhood optimization
            neighborhood_results = self.search_neighborhood_optimization(top_for_neighborhood, neighborhood_size)
        else:
            neighborhood_results = []
        
        # Phase 3: Final test evaluation
        test_results = self.final_test_evaluation(final_test_top_k)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n" + "="*80)
        print(f"🎉 ENHANCED FILTER SEARCH COMPLETE!")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"📊 Total experiments: {len(self.results)}")
        print(f"✅ Successful experiments: {len([r for r in self.results if r['status'] == 'success'])}")
        print(f"📁 Results saved to: {self.results_file}")
        print(f"📈 CSV summary: {self.csv_file}")
        
        # Show validation results
        successful_val_results = [r for r in self.results if r['status'] == 'success' and not r.get('use_test', False)]
        if successful_val_results:
            best_val = max(successful_val_results, key=lambda x: x['metrics']['ndcg20'])
            print(f"\n🏆 BEST VALIDATION CONFIGURATION:")
            config = best_val['config']
            print(f"   User: {config['user_filter_design']}({config['user_init_filter']})")
            print(f"   Item: {config['item_filter_design']}({config['item_init_filter']})")
            print(f"   Bipartite: {config['bipartite_filter_design']}({config['bipartite_init_filter']})")
            print(f"   📈 Validation NDCG@20: {best_val['metrics']['ndcg20']:.6f}")
        
        # Show test results
        if test_results:
            best_test = max(test_results, key=lambda x: x['test_metrics']['ndcg20'])
            print(f"\n🎯 BEST TEST CONFIGURATION:")
            config = best_test['config']
            print(f"   User: {config['user_filter_design']}({config['user_init_filter']})")
            print(f"   Item: {config['item_filter_design']}({config['item_init_filter']})")
            print(f"   Bipartite: {config['bipartite_filter_design']}({config['bipartite_init_filter']})")
            print(f"   📈 Test NDCG@20: {best_test['test_metrics']['ndcg20']:.6f}")
            print(f"   📈 Test Recall@20: {best_test['test_metrics']['recall20']:.6f}")
            print(f"   📈 Test Precision@20: {best_test['test_metrics']['precision20']:.6f}")
            print(f"   📊 Val-Test Diff: {best_test['val_test_diff']:+.6f}")
            
            # Print the command to reproduce best result
            print(f"\n🔧 Command to reproduce best test result:")
            cmd_parts = []
            for key, value in config.items():
                cmd_parts.append(f"--{key} {value}")
            print(f"   python main.py {' '.join(cmd_parts)}")
            
            # Show top 5 test results
            print(f"\n📊 TOP 5 TEST RESULTS:")
            sorted_test = sorted(test_results, key=lambda x: x['test_metrics']['ndcg20'], reverse=True)
            for i, result in enumerate(sorted_test[:5], 1):
                config = result['config']
                print(f"   {i}. NDCG@20: {result['test_metrics']['ndcg20']:.4f} | "
                      f"U:{config['user_filter_design']}({config['user_init_filter']}) | "
                      f"I:{config['item_filter_design']}({config['item_init_filter']}) | "
                      f"B:{config['bipartite_filter_design']}({config['bipartite_init_filter']})")
        
        print(f"="*80)
        
        return {
            'priority_results': priority_results,
            'random_results': random_results,
            'neighborhood_results': neighborhood_results,
            'test_results': test_results,
            'total_time': total_time
        }


def test_single_run():
    """Test a single run to debug output parsing"""
    print("🧪 Testing single run for debugging...")
    
    cmd = [
        'python', 'main.py', 
        '--dataset', 'ml-100k',
        '--filter', 'uib',
        '--user_filter_design', 'multiscale',
        '--user_init_filter', 'smooth',
        '--item_filter_design', 'chebyshev', 
        '--item_init_filter', 'sharp',
        '--bipartite_filter_design', 'original',
        '--bipartite_init_filter', 'smooth',
        '--epochs', '5',
        '--verbose', '1'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(f"Return code: {result.returncode}")
        print(f"Output length: {len(result.stdout)} chars")
        print(f"Error length: {len(result.stderr)} chars")
        
        if result.stdout:
            print("\n--- STDOUT (last 1000 chars) ---")
            print(result.stdout[-1000:])
            
        if result.stderr:
            print("\n--- STDERR (last 500 chars) ---")
            print(result.stderr[-500:])
            
        # Test metric parsing
        searcher = EnhancedFilterSearcher('ml-100k')
        metrics = searcher.parse_metrics(result.stdout)
        print(f"\nParsed metrics: {metrics}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Universal Spectral CF Filter Search")
    
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book'],
                       help='Dataset to search on')
    parser.add_argument('--max_random_experiments', type=int, default=200,
                       help='Maximum random experiments in comprehensive search')
    parser.add_argument('--neighborhood_size', type=int, default=50,
                       help='Neighborhood size for local optimization')
    parser.add_argument('--final_test_top_k', type=int, default=10,
                       help='Number of top configs to evaluate on test data')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Epochs for each experiment')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='search_results',
                       help='Output directory for results')
    parser.add_argument('--test', action='store_true',
                       help='Run a single test experiment for debugging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.test:
        test_single_run()
        return
    
    # Base arguments to pass to each experiment
    base_args = {
        'epochs': args.epochs,
        'lr': args.lr,
        'verbose': 1,
    }
    
    # Initialize searcher
    searcher = EnhancedFilterSearcher(
        dataset=args.dataset,
        base_args=base_args,
        output_dir=args.output_dir
    )
    
    # Run enhanced search
    results = searcher.run_enhanced_search(
        max_random_experiments=args.max_random_experiments,
        neighborhood_size=args.neighborhood_size,
        final_test_top_k=args.final_test_top_k
    )


if __name__ == "__main__":
    main()