#!/usr/bin/env python3
"""
Script to run ZipNeRF training on all NeRF synthetic scenes for 50k iterations.
This script will train on all 8 scenes from the NeRF synthetic dataset.
"""

import os
import subprocess
import time
import sys
from datetime import datetime

# NeRF synthetic dataset scenes
NERF_SCENES = [
    'lego',
    'chair', 
    'drums',
    'ficus',
    'hotdog',
    'materials',
    'mic',
    'ship'
]

def run_training_for_scene(scene, data_dir, wandb_project, base_exp_name, batch_size=4096, max_steps=50000):
    """Run training for a single scene."""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    exp_name = f"{base_exp_name}_{scene}_{timestamp}"
    
    cmd = [
        "python", "run_training.py",
        "--exp_name", exp_name,
        "--data_dir", data_dir,
        "--scene", scene,
        "--wandb_project", wandb_project,
        "--batch_size", str(batch_size),
        "--max_steps", str(max_steps)
    ]
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting training for scene: {scene}")
    print(f"📋 Experiment name: {exp_name}")
    print(f"📊 Max steps: {max_steps}")
    print(f"🎯 Batch size: {batch_size}")
    print(f"{'='*60}")
    
    # Log the command being executed
    print(f"💻 Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        # Run the training command
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Successfully completed training for {scene}")
        print(f"⏱️  Training time: {elapsed_time/3600:.2f} hours")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Training failed for {scene}")
        print(f"⏱️  Time before failure: {elapsed_time/3600:.2f} hours")
        print(f"💥 Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted for {scene}")
        return False

def main():
    """Main function to run training on all scenes."""
    # Configuration
    data_dir = "/home/nilkel/Projects/data/nerf_synthetic"
    wandb_project = "my-nerf-experiments"
    base_exp_name = "lego_triplane_relu"
    batch_size = 4096
    max_steps = 50000
    
    print("🔥 ZipNeRF Triplane Training - All NeRF Synthetic Scenes")
    print("="*70)
    print(f"📁 Data directory: {data_dir}")
    print(f"📊 Wandb project: {wandb_project}")
    print(f"🏷️  Base experiment name: {base_exp_name}")
    print(f"🎯 Batch size: {batch_size}")
    print(f"📈 Max steps per scene: {max_steps}")
    print(f"🎬 Total scenes: {len(NERF_SCENES)}")
    print(f"📝 Scenes: {', '.join(NERF_SCENES)}")
    print("="*70)
    
    # Verify data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory does not exist: {data_dir}")
        print("Please update the data_dir variable in this script.")
        sys.exit(1)
    
    # Check that at least some scenes exist
    existing_scenes = []
    for scene in NERF_SCENES:
        scene_path = os.path.join(data_dir, scene)
        if os.path.exists(scene_path):
            existing_scenes.append(scene)
        else:
            print(f"⚠️  Warning: Scene directory not found: {scene_path}")
    
    if not existing_scenes:
        print("❌ Error: No valid scene directories found!")
        sys.exit(1)
    
    print(f"✅ Found {len(existing_scenes)} valid scenes: {', '.join(existing_scenes)}")
    
    # Ask for confirmation
    total_estimated_time = len(existing_scenes) * 3  # Rough estimate: 3 hours per scene
    print(f"\n⏱️  Estimated total time: ~{total_estimated_time} hours")
    response = input(f"\n🤖 Start training on {len(existing_scenes)} scenes? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("Training cancelled.")
        sys.exit(0)
    
    # Run training for each scene
    successful_scenes = []
    failed_scenes = []
    start_time = time.time()
    
    for i, scene in enumerate(existing_scenes, 1):
        print(f"\n🎯 Scene {i}/{len(existing_scenes)}: {scene}")
        
        success = run_training_for_scene(
            scene=scene,
            data_dir=data_dir,
            wandb_project=wandb_project,
            base_exp_name=base_exp_name,
            batch_size=batch_size,
            max_steps=max_steps
        )
        
        if success:
            successful_scenes.append(scene)
        else:
            failed_scenes.append(scene)
            
            # Ask if we should continue after a failure
            response = input(f"\n⚠️  Continue with remaining scenes? [Y/n]: ").strip().lower()
            if response and response not in ['y', 'yes']:
                print("Training stopped by user.")
                break
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("📊 TRAINING SUMMARY")
    print("="*70)
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"✅ Successful scenes ({len(successful_scenes)}): {', '.join(successful_scenes)}")
    if failed_scenes:
        print(f"❌ Failed scenes ({len(failed_scenes)}): {', '.join(failed_scenes)}")
    else:
        print("🎉 All scenes completed successfully!")
    print("="*70)
    
    if failed_scenes:
        print("\n💡 To retry failed scenes, you can run individual commands:")
        for scene in failed_scenes:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            exp_name = f"{base_exp_name}_{scene}_{timestamp}"
            print(f"   python run_training.py --exp_name {exp_name} --data_dir {data_dir} --scene {scene} --wandb_project {wandb_project} --batch_size {batch_size} --max_steps {max_steps}")

if __name__ == "__main__":
    main() 