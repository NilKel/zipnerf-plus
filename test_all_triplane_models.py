#!/usr/bin/env python3
"""
Comprehensive test script for all trained ZipNeRF triplane models.

This script automatically discovers all triplane experiments, validates their checkpoints,
and runs evaluation on their respective test sets. It generates detailed reports with
metrics comparison across all models.

Usage:
    python test_all_triplane_models.py [options]
    
Options:
    --dry_run          : Only discover and validate models, don't run evaluation
    --scenes SCENES    : Comma-separated list of scenes to test (e.g., "lego,drums")
    --config CONFIG    : Path to gin config file (default: configs/blender.gin)
    --output OUTPUT    : Output directory for reports (default: current directory)
    
Examples:
    python test_all_triplane_models.py --dry_run
    python test_all_triplane_models.py --scenes lego,drums
    python test_all_triplane_models.py --config configs/llff.gin
"""

import os
import sys
import glob
import json
import subprocess
import time
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Try to import pandas for better table formatting
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not installed. CSV export and table formatting will be limited.")

class TriplaneModelTester:
    """Comprehensive tester for ZipNeRF triplane models."""
    
    def __init__(self, config_path: str = "configs/blender.gin", output_dir: str = "."):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.exp_dir = Path("exp")
        self.data_base_dir = Path("/home/nilkel/Projects/data/nerf_synthetic")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results: List[Dict[str, Any]] = []
        
    def activate_environment(self) -> bool:
        """Activate the zipnerf environment by sourcing the activation script."""
        try:
            # Check if we're already in the right environment
            result = subprocess.run(
                ["python", "-c", "import torch; print('torch available')"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return True
                
            # Try to activate environment
            activate_script = Path("activate_zipnerf.sh")
            if activate_script.exists():
                print("🔧 Activating ZipNeRF environment...")
                # We'll need to modify commands to include environment activation
                return True
            else:
                print("⚠️  Warning: activate_zipnerf.sh not found. Assuming environment is ready.")
                return True
                
        except Exception as e:
            print(f"⚠️  Warning: Could not verify environment: {e}")
            return True  # Continue anyway
    
    def find_triplane_experiments(self) -> List[Dict[str, Any]]:
        """Find all triplane experiments and their status."""
        if not self.exp_dir.exists():
            print(f"❌ Experiment directory {self.exp_dir} not found!")
            return []
        
        # Find all directories matching triplane pattern
        triplane_pattern = "lego_triplane_relu_*"
        exp_dirs = list(self.exp_dir.glob(triplane_pattern))
        
        if not exp_dirs:
            print(f"❌ No triplane experiments found matching pattern: {triplane_pattern}")
            return []
        
        experiments = []
        for exp_path in sorted(exp_dirs):
            exp_name = exp_path.name
            
            # Extract scene name from experiment name
            scene_name = self._extract_scene_name(exp_name)
            
            # Check for checkpoints
            checkpoints_dir = exp_path / "checkpoints"
            if not checkpoints_dir.exists():
                status = "❌ Not ready"
                detail = "No checkpoints directory"
                latest_checkpoint = None
            else:
                # Look for checkpoint directories (like 025000/, 050000/)
                checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                if not checkpoint_dirs:
                    status = "❌ Not ready"
                    detail = "No checkpoint directories"
                    latest_checkpoint = None
                else:
                    # Find latest checkpoint by directory name
                    checkpoint_numbers = [int(d.name) for d in checkpoint_dirs]
                    latest_step = max(checkpoint_numbers)
                    latest_checkpoint = f"{latest_step:06d}"
                    status = "✅ Ready"
                    detail = f"Latest checkpoint: {latest_checkpoint}"
            
            # Determine data directory
            data_dir = self.data_base_dir / scene_name
            
            experiments.append({
                'exp_name': exp_name,
                'exp_path': exp_path,
                'scene_name': scene_name,
                'data_dir': str(data_dir),
                'status': status,
                'detail': detail,
                'latest_checkpoint': latest_checkpoint,
                'ready': latest_checkpoint is not None
            })
        
        return experiments
    
    def _extract_scene_name(self, exp_name: str) -> str:
        """Extract scene name from experiment name."""
        # Handle different naming patterns
        
        # Pattern 1: lego_triplane_relu_SCENE_...
        # Look for known scene names
        known_scenes = ['lego', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'ship', 'chair']
        
        # First check if it's a simple lego experiment
        if exp_name.startswith('lego_triplane_relu_') and not any(scene in exp_name[19:] for scene in known_scenes[1:]):
            return 'lego'
        
        # Then check for other scenes
        for scene in known_scenes[1:]:  # Skip lego as we handled it above
            if f'_{scene}_' in exp_name:
                return scene
        
        # Default fallback
        return 'lego'
    
    def run_evaluation(self, exp_info: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        """Run evaluation for a single experiment."""
        exp_name = exp_info['exp_name']
        data_dir = exp_info['data_dir']
        scene_name = exp_info['scene_name']
        
        print(f"\n{'='*60}")
        print(f"🧪 EVALUATING: {exp_name}")
        print(f"📁 Data dir: {data_dir}")
        
        if dry_run:
            print("🔍 DRY RUN MODE - Skipping actual evaluation")
            return {
                'exp_name': exp_name,
                'scene_name': scene_name,
                'data_dir': data_dir,
                'success': True,
                'message': 'Dry run completed',
                'duration': '0.0s',
                'metrics': {},
                'timestamp': datetime.now().isoformat()
            }
        
        # Build evaluation command with environment activation - fix gin binding format
        cmd_str = f'source activate_zipnerf.sh && accelerate launch eval.py --gin_configs={self.config_path} --gin_bindings="Config.data_dir = \'{data_dir}\'" --gin_bindings="Config.exp_name = \'{exp_name}\'" --gin_bindings="Config.eval_only_once = True" --gin_bindings="Config.eval_save_output = True"'
        cmd = ['bash', '-c', cmd_str]
        print(f"💻 Command: {cmd_str}")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Run evaluation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            duration = time.time() - start_time
            duration_str = f"{duration:.1f}s"
            
            if result.returncode == 0:
                # Parse metrics from output
                metrics = self._parse_evaluation_metrics(result.stdout)
                
                return {
                    'exp_name': exp_name,
                    'scene_name': scene_name,
                    'data_dir': data_dir,
                    'success': True,
                    'message': 'Evaluation completed successfully',
                    'duration': duration_str,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat(),
                    'stdout': result.stdout[-1000:],  # Last 1000 chars for debugging
                }
            else:
                return {
                    'exp_name': exp_name,
                    'scene_name': scene_name,
                    'data_dir': data_dir,
                    'success': False,
                    'message': f'Evaluation failed with return code {result.returncode}',
                    'duration': duration_str,
                    'metrics': {},
                    'timestamp': datetime.now().isoformat(),
                    'stderr': result.stderr[-1000:],  # Last 1000 chars for debugging
                }
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                'exp_name': exp_name,
                'scene_name': scene_name,
                'data_dir': data_dir,
                'success': False,
                'message': 'Evaluation timed out after 1 hour',
                'duration': f"{duration:.1f}s",
                'metrics': {},
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            return {
                'exp_name': exp_name,
                'scene_name': scene_name,
                'data_dir': data_dir,
                'success': False,
                'message': f'Evaluation failed with exception: {e}',
                'duration': f"{duration:.1f}s",
                'metrics': {},
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_evaluation_metrics(self, output: str) -> Dict[str, float]:
        """Parse evaluation metrics from output."""
        metrics = {}
        
        # Look for metrics in the output
        lines = output.split('\n')
        for line in lines:
            # Look for patterns like "PSNR: 25.34" or "test_psnr 25.34"
            psnr_match = re.search(r'(?:PSNR|test_psnr).*?(\d+\.\d+)', line, re.IGNORECASE)
            if psnr_match:
                metrics['psnr'] = float(psnr_match.group(1))
            
            ssim_match = re.search(r'(?:SSIM|test_ssim).*?(\d+\.\d+)', line, re.IGNORECASE)
            if ssim_match:
                metrics['ssim'] = float(ssim_match.group(1))
            
            lpips_match = re.search(r'(?:LPIPS|test_lpips).*?(\d+\.\d+)', line, re.IGNORECASE)
            if lpips_match:
                metrics['lpips'] = float(lpips_match.group(1))
        
        return metrics
    
    def filter_experiments_by_scenes(self, experiments: List[Dict], scenes: List[str]) -> List[Dict]:
        """Filter experiments to only include specified scenes."""
        if not scenes:
            return experiments
        
        filtered = []
        for exp in experiments:
            if exp['scene_name'] in scenes and exp['ready']:
                filtered.append(exp)
        
        return filtered
    
    def run_all_evaluations(self, scenes: Optional[List[str]] = None, dry_run: bool = False):
        """Run evaluations for all triplane models."""
        print("🔍 Finding triplane experiments...")
        
        # Activate environment first
        if not self.activate_environment():
            print("❌ Failed to activate environment")
            return
        
        experiments = self.find_triplane_experiments()
        
        if not experiments:
            print("❌ No triplane experiments found!")
            return
        
        print(f"✅ Found {len(experiments)} triplane experiments:")
        for exp in experiments:
            scene_name = exp['scene_name']
            exp_name = exp['exp_name']
            status = exp['status']
            detail = exp['detail']
            print(f"  {scene_name:<12} | {exp_name:<50} | {status} ({detail})")
        
        # Filter by scenes if specified
        if scenes:
            experiments = self.filter_experiments_by_scenes(experiments, scenes)
            print(f"\n🎯 Filtered to {len(experiments)} experiments for scenes: {', '.join(scenes)}")
        
        # Filter to only ready experiments
        ready_experiments = [exp for exp in experiments if exp['ready']]
        skipped_experiments = [exp for exp in experiments if not exp['ready']]
        
        if skipped_experiments:
            print(f"\n⚠️  Skipping {len(skipped_experiments)} experiments:")
            for exp in skipped_experiments:
                print(f"     {exp['exp_name']}: {exp['detail']}")
        
        if not ready_experiments:
            print("❌ No ready experiments found!")
            return
        
        print(f"\n🚀 Will evaluate {len(ready_experiments)} ready experiments")
        
        if not dry_run:
            estimated_time = len(ready_experiments) * 10  # Rough estimate: 10 min per model
            print(f"⏱️  Estimated time: {estimated_time//60}h {estimated_time%60}m")
            print("🏃 Starting evaluation automatically...")
        
        # Run evaluations
        total_start_time = time.time()
        
        for i, exp in enumerate(ready_experiments, 1):
            print(f"\n📋 Progress: {i}/{len(ready_experiments)}")
            
            result = self.run_evaluation(exp, dry_run=dry_run)
            self.results.append(result)
            
            if result['success']:
                print(f"✅ SUCCESS: {result['scene_name']}")
                if result['metrics']:
                    metrics = result['metrics']
                    print(f"   PSNR: {metrics.get('psnr', 'N/A'):.2f}" if 'psnr' in metrics else "   PSNR: N/A")
                    print(f"   SSIM: {metrics.get('ssim', 'N/A'):.4f}" if 'ssim' in metrics else "   SSIM: N/A")
                    print(f"   LPIPS: {metrics.get('lpips', 'N/A'):.4f}" if 'lpips' in metrics else "   LPIPS: N/A")
            else:
                print(f"❌ FAILED: {result['scene_name']}")
                print(f"   Error: {result['message']}")
        
        total_duration = time.time() - total_start_time
        print(f"\n⏱️  Total evaluation time: {total_duration:.1f}s")
        
        # Generate reports
        self.generate_reports()
    
    def generate_reports(self):
        """Generate JSON and CSV reports."""
        # Save detailed JSON report
        json_path = self.output_dir / "triplane_evaluation_report.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary statistics
        summary = self._generate_summary()
        
        # Print summary to console
        self._print_summary(summary)
        
        # Save CSV report if pandas is available
        csv_path = self.output_dir / "triplane_evaluation_report.csv"
        self._save_csv_report(csv_path, summary)
        
        print(f"\n💾 Detailed report saved to: {json_path}")
        print(f"💾 CSV summary saved to: {csv_path}")
        print("="*80)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_experiments = len(self.results)
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        summary = {
            'total_experiments': total_experiments,
            'successful_evaluations': len(successful),
            'failed_evaluations': len(failed),
            'results': []
        }
        
        # Collect metrics by scene
        scene_metrics = {}
        for result in successful:
            scene = result['scene_name']
            metrics = result['metrics']
            
            if scene not in scene_metrics:
                scene_metrics[scene] = []
            
            scene_metrics[scene].append({
                'exp_name': result['exp_name'],
                'psnr': metrics.get('psnr'),
                'ssim': metrics.get('ssim'),
                'lpips': metrics.get('lpips'),
                'duration': result['duration']
            })
        
        # Add results for table
        for result in self.results:
            metrics = result['metrics']
            summary['results'].append({
                'scene_name': result['scene_name'],
                'exp_name': result['exp_name'],
                'status': 'Success' if result['success'] else 'Failed',
                'psnr': metrics.get('psnr') if metrics else None,
                'ssim': metrics.get('ssim') if metrics else None,
                'lpips': metrics.get('lpips') if metrics else None,
                'duration': result['duration']
            })
        
        summary['scene_metrics'] = scene_metrics
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print summary to console."""
        print("\n" + "="*80)
        print("📊 TRIPLANE MODELS EVALUATION SUMMARY")
        print("="*80)
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Successful evaluations: {summary['successful_evaluations']}")
        print(f"Failed evaluations: {summary['failed_evaluations']}")
        
        if summary['successful_evaluations'] > 0:
            print("\n📈 METRICS SUMMARY:")
            
            # Calculate overall statistics
            all_psnr = [r['psnr'] for r in summary['results'] if r['psnr'] is not None]
            all_ssim = [r['ssim'] for r in summary['results'] if r['ssim'] is not None]
            all_lpips = [r['lpips'] for r in summary['results'] if r['lpips'] is not None]
            
            if all_psnr:
                print(f"PSNR  - Mean: {sum(all_psnr)/len(all_psnr):.2f}, Min: {min(all_psnr):.2f}, Max: {max(all_psnr):.2f}")
            if all_ssim:
                print(f"SSIM  - Mean: {sum(all_ssim)/len(all_ssim):.4f}, Min: {min(all_ssim):.4f}, Max: {max(all_ssim):.4f}")
            if all_lpips:
                print(f"LPIPS - Mean: {sum(all_lpips)/len(all_lpips):.4f}, Min: {min(all_lpips):.4f}, Max: {max(all_lpips):.4f}")
        
        # Print detailed table
        print("\n📈 METRICS SUMMARY:")
        self._print_results_table(summary['results'])
    
    def _print_results_table(self, results: List[Dict]):
        """Print results in a formatted table."""
        if PANDAS_AVAILABLE:
            # Use pandas for nice formatting
            df = pd.DataFrame(results)
            df = df[['scene_name', 'exp_name', 'status', 'psnr', 'ssim', 'lpips', 'duration']]
            df.columns = ['Scene', 'Experiment', 'Status', 'PSNR', 'SSIM', 'LPIPS', 'Duration']
            
            # Truncate long experiment names for display
            df['Experiment'] = df['Experiment'].apply(lambda x: x[:35] + '...' if len(x) > 38 else x)
            
            print(df.to_string(index=False, float_format='%.2f'))
        else:
            # Manual table formatting
            header = f"{'Scene':<11} | {'Experiment':<43} | {'Status':<6} | {'PSNR':<6} | {'SSIM':<6} | {'LPIPS':<6} | {'Duration':<8}"
            print(header)
            print("-" * len(header))
            
            for result in results:
                exp_name = result['exp_name']
                if len(exp_name) > 40:
                    exp_name = exp_name[:37] + "..."
                
                psnr = f"{result['psnr']:.2f}" if result['psnr'] is not None else "N/A"
                ssim = f"{result['ssim']:.3f}" if result['ssim'] is not None else "N/A"
                lpips = f"{result['lpips']:.3f}" if result['lpips'] is not None else "N/A"
                
                row = f"{result['scene_name']:<11} | {exp_name:<43} | {result['status']:<6} | {psnr:<6} | {ssim:<6} | {lpips:<6} | {result['duration']:<8}"
                print(row)
    
    def _save_csv_report(self, csv_path: Path, summary: Dict[str, Any]):
        """Save CSV report."""
        try:
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(summary['results'])
                df.to_csv(csv_path, index=False)
            else:
                # Manual CSV writing
                with open(csv_path, 'w') as f:
                    f.write("scene_name,exp_name,status,psnr,ssim,lpips,duration\n")
                    for result in summary['results']:
                        f.write(f"{result['scene_name']},{result['exp_name']},{result['status']},")
                        f.write(f"{result['psnr'] or 'N/A'},{result['ssim'] or 'N/A'},{result['lpips'] or 'N/A'},")
                        f.write(f"{result['duration']}\n")
                        
        except Exception as e:
            print(f"⚠️  Warning: Could not save CSV report: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test script for ZipNeRF triplane models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry_run                    # Discover models without evaluation
  %(prog)s --scenes lego,drums          # Test only specific scenes
  %(prog)s --config configs/llff.gin    # Use different config
        """
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Only discover and validate models without running evaluation'
    )
    
    parser.add_argument(
        '--scenes',
        type=str,
        help='Comma-separated list of scenes to test (e.g., "lego,drums")'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/blender.gin',
        help='Path to gin config file (default: configs/blender.gin)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Output directory for reports (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Parse scenes
    scenes = None
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(',')]
    
    # Create tester and run
    tester = TriplaneModelTester(config_path=args.config, output_dir=args.output)
    tester.run_all_evaluations(scenes=scenes, dry_run=args.dry_run)


if __name__ == "__main__":
    main() 