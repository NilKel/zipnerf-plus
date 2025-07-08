#!/usr/bin/env python3
"""
Comprehensive ZipNeRF Triplane Model Organization and Testing Script

This script will:
1. Organize all triplane models into a clean directory structure
2. Rename models by removing the "lego_" prefix
3. Run full evaluation on the entire Blender synthetic dataset
4. Extract comprehensive metrics and generate reports

Usage:
    python organize_and_test_triplane_models.py [--dry_run] [--organize_only] [--test_only]
"""

import os
import sys
import shutil
import subprocess
import json
import csv
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

# Blender synthetic dataset scenes
BLENDER_SCENES = ["drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

class TriplaneModelOrganizer:
    """Organizes and tests ZipNeRF triplane models."""
    
    def __init__(self, base_data_dir: str = "/home/nilkel/Projects/data/nerf_synthetic"):
        self.base_data_dir = Path(base_data_dir)
        self.exp_dir = Path("exp")
        self.organized_dir = Path("exp/triplane_models")
        self.results_dir = Path("triplane_results")
        
        # Create directories
        self.organized_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: List[Dict[str, Any]] = []
        
    def find_triplane_models(self) -> List[Dict[str, Any]]:
        """Find all triplane models in the experiment directory."""
        models = []
        
        if not self.exp_dir.exists():
            print(f"❌ Experiment directory not found: {self.exp_dir}")
            return models
            
        # Look for triplane models
        for exp_path in self.exp_dir.iterdir():
            if exp_path.is_dir() and "triplane_relu" in exp_path.name:
                # Extract scene name from experiment name
                scene_name = self._extract_scene_name(exp_path.name)
                if scene_name:
                    checkpoint_dir = exp_path / "checkpoints"
                    if checkpoint_dir.exists():
                        # Find available checkpoints
                        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                        if checkpoints:
                            # Use the highest checkpoint
                            latest_checkpoint = max(checkpoints, key=lambda x: int(x.name))
                            
                            models.append({
                                'original_name': exp_path.name,
                                'scene_name': scene_name,
                                'original_path': exp_path,
                                'checkpoint_path': latest_checkpoint,
                                'checkpoint_step': int(latest_checkpoint.name),
                                'new_name': f"triplane_relu_{scene_name}",
                                'data_dir': self.base_data_dir / scene_name
                            })
                        else:
                            print(f"⚠️  No checkpoints found for {exp_path.name}")
                    else:
                        print(f"⚠️  No checkpoint directory found for {exp_path.name}")
        
        return models
    
    def _extract_scene_name(self, exp_name: str) -> Optional[str]:
        """Extract scene name from experiment name."""
        # Remove lego_ prefix and extract scene name
        patterns = [
            r'lego_triplane_relu_(\w+)_\d+_\d+_\d+_\d+',
            r'triplane_relu_(\w+)_\d+_\d+_\d+_\d+',
            r'lego_triplane_relu_(\w+)',
            r'triplane_relu_(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, exp_name)
            if match:
                scene_name = match.group(1)
                if scene_name in BLENDER_SCENES:
                    return scene_name
        
        return None
    
    def organize_models(self, models: List[Dict[str, Any]], dry_run: bool = False) -> List[Dict[str, Any]]:
        """Organize models into clean directory structure."""
        organized_models = []
        
        print(f"\n🗂️  ORGANIZING {len(models)} TRIPLANE MODELS")
        print("=" * 60)
        
        for model in models:
            new_path = self.organized_dir / model['new_name']
            
            print(f"📁 {model['original_name']} -> {model['new_name']}")
            print(f"   Source: {model['original_path']}")
            print(f"   Target: {new_path}")
            print(f"   Checkpoint: {model['checkpoint_step']} steps")
            
            if not dry_run:
                if new_path.exists():
                    print(f"   ⚠️  Target already exists, skipping...")
                else:
                    try:
                        shutil.copytree(model['original_path'], new_path)
                        print(f"   ✅ Successfully organized")
                    except Exception as e:
                        print(f"   ❌ Error organizing: {e}")
                        continue
            else:
                print(f"   🔍 DRY RUN - Would organize")
            
            organized_models.append({
                **model,
                'organized_path': new_path,
                'organized_checkpoint': new_path / "checkpoints" / str(model['checkpoint_step']),
                'organized': not dry_run
            })
            print()
        
        return organized_models
    
    def run_evaluation(self, model: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        """Run evaluation for a single model."""
        scene_name = model['scene_name']
        exp_name = model['new_name']
        data_dir = model['data_dir']
        
        print(f"\n🧪 EVALUATING: {exp_name}")
        print(f"📁 Scene: {scene_name}")
        print(f"📁 Data: {data_dir}")
        print(f"🔢 Checkpoint: {model['checkpoint_step']} steps")
        
        if not data_dir.exists():
            return {
                'exp_name': exp_name,
                'scene_name': scene_name,
                'success': False,
                'error': f"Data directory not found: {data_dir}",
                'metrics': {}
            }
        
        if dry_run:
            print("🔍 DRY RUN - Skipping evaluation")
            return {
                'exp_name': exp_name,
                'scene_name': scene_name,
                'success': True,
                'message': 'Dry run completed',
                'metrics': {'psnr': 25.0, 'ssim': 0.85, 'lpips': 0.15}  # Mock metrics
            }
        
        # Use organized path if available, otherwise original
        model_path = model.get('organized_path', model['original_path'])
        
        # Build evaluation command
        cmd_parts = [
            'source activate_zipnerf.sh &&',
            'accelerate launch eval.py',
            '--gin_configs=configs/blender.gin',
            f'--gin_bindings="Config.data_dir = \'{data_dir}\'"',
            f'--gin_bindings="Config.exp_name = \'{model_path.relative_to(Path("exp"))}\'"',
            '--gin_bindings="Config.eval_only_once = True"',
            '--gin_bindings="Config.eval_save_output = True"'
        ]
        
        cmd_str = ' '.join(cmd_parts)
        cmd = ['bash', '-c', cmd_str]
        
        print(f"💻 Command: {cmd_str}")
        print("=" * 60)
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=os.getcwd()
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Evaluation completed successfully in {duration:.1f}s")
                
                # Extract metrics from output
                metrics = self._extract_metrics_from_output(result.stdout, result.stderr)
                
                return {
                    'exp_name': exp_name,
                    'scene_name': scene_name,
                    'success': True,
                    'duration': f"{duration:.1f}s",
                    'metrics': metrics,
                    'checkpoint_step': model['checkpoint_step'],
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                print(f"❌ Evaluation failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                
                return {
                    'exp_name': exp_name,
                    'scene_name': scene_name,
                    'success': False,
                    'error': f"Return code {result.returncode}",
                    'duration': f"{duration:.1f}s",
                    'metrics': {},
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Evaluation timed out after 1 hour")
            return {
                'exp_name': exp_name,
                'scene_name': scene_name,
                'success': False,
                'error': "Timeout after 1 hour",
                'metrics': {}
            }
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {
                'exp_name': exp_name,
                'scene_name': scene_name,
                'success': False,
                'error': str(e),
                'metrics': {}
            }
    
    def _extract_metrics_from_output(self, stdout: str, stderr: str) -> Dict[str, float]:
        """Extract PSNR, SSIM, LPIPS metrics from evaluation output."""
        metrics = {}
        
        # Look for metrics in both stdout and stderr
        combined_output = stdout + stderr
        
        # Common patterns for metrics
        patterns = {
            'psnr': [
                r'PSNR:\s*([0-9.]+)',
                r'psnr.*?([0-9.]+)',
                r'Peak Signal-to-Noise Ratio.*?([0-9.]+)'
            ],
            'ssim': [
                r'SSIM:\s*([0-9.]+)',
                r'ssim.*?([0-9.]+)',
                r'Structural Similarity.*?([0-9.]+)'
            ],
            'lpips': [
                r'LPIPS:\s*([0-9.]+)',
                r'lpips.*?([0-9.]+)',
                r'Learned Perceptual.*?([0-9.]+)'
            ]
        }
        
        for metric_name, metric_patterns in patterns.items():
            for pattern in metric_patterns:
                matches = re.findall(pattern, combined_output, re.IGNORECASE)
                if matches:
                    try:
                        # Take the last match (usually the final result)
                        metrics[metric_name] = float(matches[-1])
                        break
                    except ValueError:
                        continue
        
        return metrics
    
    def run_full_evaluation(self, models: List[Dict[str, Any]], dry_run: bool = False) -> None:
        """Run evaluation on all models."""
        print(f"\n🚀 RUNNING FULL EVALUATION ON {len(models)} MODELS")
        print("=" * 70)
        
        successful_evals = 0
        failed_evals = 0
        
        for i, model in enumerate(models, 1):
            print(f"\n📊 PROGRESS: {i}/{len(models)} models")
            
            result = self.run_evaluation(model, dry_run)
            self.results.append(result)
            
            if result['success']:
                successful_evals += 1
                if result.get('metrics'):
                    print(f"📈 Metrics: {result['metrics']}")
            else:
                failed_evals += 1
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        print(f"\n📊 EVALUATION SUMMARY")
        print("=" * 30)
        print(f"✅ Successful: {successful_evals}")
        print(f"❌ Failed: {failed_evals}")
        print(f"📊 Total: {len(models)}")
        
        # Generate reports
        self._generate_reports()
    
    def _generate_reports(self) -> None:
        """Generate comprehensive reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_file = self.results_dir / f"triplane_evaluation_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 JSON report saved: {json_file}")
        
        # CSV report
        csv_file = self.results_dir / f"triplane_evaluation_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'exp_name', 'scene_name', 'success', 'checkpoint_step',
                'psnr', 'ssim', 'lpips', 'duration', 'error'
            ])
            writer.writeheader()
            
            for result in self.results:
                metrics = result.get('metrics', {})
                writer.writerow({
                    'exp_name': result['exp_name'],
                    'scene_name': result['scene_name'],
                    'success': result['success'],
                    'checkpoint_step': result.get('checkpoint_step', ''),
                    'psnr': metrics.get('psnr', ''),
                    'ssim': metrics.get('ssim', ''),
                    'lpips': metrics.get('lpips', ''),
                    'duration': result.get('duration', ''),
                    'error': result.get('error', '')
                })
        
        print(f"📊 CSV report saved: {csv_file}")
        
        # Summary report
        summary_file = self.results_dir / f"triplane_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("ZipNeRF Triplane Models Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            successful_results = [r for r in self.results if r['success']]
            
            if successful_results:
                f.write("SUCCESSFUL EVALUATIONS:\n")
                f.write("-" * 25 + "\n")
                
                for result in successful_results:
                    f.write(f"Scene: {result['scene_name']}\n")
                    f.write(f"Model: {result['exp_name']}\n")
                    f.write(f"Checkpoint: {result.get('checkpoint_step', 'N/A')} steps\n")
                    
                    metrics = result.get('metrics', {})
                    if metrics:
                        f.write(f"PSNR: {metrics.get('psnr', 'N/A'):.3f}\n")
                        f.write(f"SSIM: {metrics.get('ssim', 'N/A'):.3f}\n")
                        f.write(f"LPIPS: {metrics.get('lpips', 'N/A'):.3f}\n")
                    
                    f.write(f"Duration: {result.get('duration', 'N/A')}\n")
                    f.write("\n")
                
                # Calculate averages
                if successful_results:
                    avg_psnr = sum(r.get('metrics', {}).get('psnr', 0) for r in successful_results) / len(successful_results)
                    avg_ssim = sum(r.get('metrics', {}).get('ssim', 0) for r in successful_results) / len(successful_results)
                    avg_lpips = sum(r.get('metrics', {}).get('lpips', 0) for r in successful_results) / len(successful_results)
                    
                    f.write("AVERAGE METRICS:\n")
                    f.write("-" * 16 + "\n")
                    f.write(f"Average PSNR: {avg_psnr:.3f}\n")
                    f.write(f"Average SSIM: {avg_ssim:.3f}\n")
                    f.write(f"Average LPIPS: {avg_lpips:.3f}\n")
            
            failed_results = [r for r in self.results if not r['success']]
            if failed_results:
                f.write("\nFAILED EVALUATIONS:\n")
                f.write("-" * 20 + "\n")
                for result in failed_results:
                    f.write(f"Scene: {result['scene_name']}\n")
                    f.write(f"Model: {result['exp_name']}\n")
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n\n")
        
        print(f"📝 Summary report saved: {summary_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Organize and test ZipNeRF triplane models")
    parser.add_argument("--dry_run", action="store_true", help="Run in dry-run mode (no actual changes)")
    parser.add_argument("--organize_only", action="store_true", help="Only organize models, don't run evaluation")
    parser.add_argument("--test_only", action="store_true", help="Only run evaluation, don't organize")
    parser.add_argument("--data_dir", default="/home/nilkel/Projects/data/nerf_synthetic", help="Base data directory")
    
    args = parser.parse_args()
    
    print("🚀 ZipNeRF Triplane Model Organizer and Tester")
    print("=" * 60)
    
    organizer = TriplaneModelOrganizer(args.data_dir)
    
    # Find all triplane models
    models = organizer.find_triplane_models()
    
    if not models:
        print("❌ No triplane models found!")
        return
    
    print(f"🔍 Found {len(models)} triplane models:")
    for model in models:
        print(f"  • {model['scene_name']}: {model['original_name']} ({model['checkpoint_step']} steps)")
    
    # Organize models (unless test_only)
    if not args.test_only:
        organized_models = organizer.organize_models(models, args.dry_run)
    else:
        organized_models = models
    
    # Run evaluation (unless organize_only)
    if not args.organize_only:
        organizer.run_full_evaluation(organized_models, args.dry_run)
    
    print("\n🎉 Script completed!")


if __name__ == "__main__":
    main()