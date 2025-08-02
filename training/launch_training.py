#!/usr/bin/env python3
"""
Training Launcher for Crisis Response AI

Simple launcher script to train individual algorithms with optimized configurations.
Provides easy command-line interface for training any of the 4 RL algorithms.

Usage Examples:
    # Train DQN with default settings
    python launch_training.py dqn
    
    # Train PPO with visualization and recording
    python launch_training.py ppo --visualize --record
    
    # Train all algorithms sequentially
    python launch_training.py all --episodes 500
    
    # Quick test run
    python launch_training.py reinforce --episodes 100 --variant fast
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def launch_dqn(args):
    """Launch DQN training"""
    cmd = [
        sys.executable, 
        str(project_root / "training" / "train_dqn.py"),
        "--episodes", str(args.episodes),
        "--save-dir", f"models/dqn_{int(time.time())}"
    ]
    
    if args.visualize:
        cmd.append("--visualize")
    if args.record:
        cmd.append("--record")
    if args.gpu:
        cmd.append("--gpu")
    if args.config:
        cmd.extend(["--config", args.config])
    
    return subprocess.run(cmd)

def launch_reinforce(args):
    """Launch REINFORCE training"""
    cmd = [
        sys.executable,
        str(project_root / "training" / "train_reinforce.py"),
        "--episodes", str(args.episodes),
        "--save-dir", f"models/reinforce_{int(time.time())}"
    ]
    
    if args.visualize:
        cmd.append("--visualize")
    if args.record:
        cmd.append("--record")
    if args.gpu:
        cmd.append("--gpu")
    if args.config:
        cmd.extend(["--config", args.config])
    if args.trajectories:
        cmd.extend(["--trajectories", str(args.trajectories)])
    
    return subprocess.run(cmd)

def launch_ppo(args):
    """Launch PPO training"""
    cmd = [
        sys.executable,
        str(project_root / "training" / "train_ppo.py"),
        "--episodes", str(args.episodes),
        "--save-dir", f"models/ppo_{int(time.time())}"
    ]
    
    if args.visualize:
        cmd.append("--visualize")
    if args.record:
        cmd.append("--record")
    if args.gpu:
        cmd.append("--gpu")
    if args.config:
        cmd.extend(["--config", args.config])
    if args.steps:
        cmd.extend(["--steps", str(args.steps)])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    
    return subprocess.run(cmd)

def launch_a2c(args):
    """Launch A2C training"""
    cmd = [
        sys.executable,
        str(project_root / "training" / "train_a2c.py"),
        "--episodes", str(args.episodes),
        "--save-dir", f"models/a2c_{int(time.time())}"
    ]
    
    if args.visualize:
        cmd.append("--visualize")
    if args.record:
        cmd.append("--record")
    if args.gpu:
        cmd.append("--gpu")
    if args.config:
        cmd.extend(["--config", args.config])
    if args.n_steps:
        cmd.extend(["--n-steps", str(args.n_steps)])
    if args.shared:
        cmd.append("--shared")
    if args.rms_prop:
        cmd.append("--rms-prop")
    
    return subprocess.run(cmd)

def launch_all(args):
    """Launch all algorithms sequentially"""
    algorithms = ['dqn', 'reinforce', 'ppo', 'a2c']
    results = {}
    
    print("🚀 LAUNCHING ALL ALGORITHMS SEQUENTIALLY")
    print("=" * 60)
    
    for algo in algorithms:
        print(f"\n🔄 Starting {algo.upper()} training...")
        start_time = time.time()
        
        # Update args for current algorithm
        args.algorithm = algo
        
        # Launch algorithm
        if algo == 'dqn':
            result = launch_dqn(args)
        elif algo == 'reinforce':
            result = launch_reinforce(args)
        elif algo == 'ppo':
            result = launch_ppo(args)
        elif algo == 'a2c':
            result = launch_a2c(args)
        
        training_time = time.time() - start_time
        results[algo] = {
            'return_code': result.returncode,
            'training_time': training_time,
            'success': result.returncode == 0
        }
        
        if result.returncode == 0:
            print(f"✅ {algo.upper()} completed successfully in {training_time:.2f}s")
        else:
            print(f"❌ {algo.upper()} failed with code {result.returncode}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    
    total_time = sum(r['training_time'] for r in results.values())
    successful = [algo for algo, r in results.items() if r['success']]
    failed = [algo for algo, r in results.items() if not r['success']]
    
    print(f"✅ Successful: {', '.join(successful) if successful else 'None'}")
    print(f"❌ Failed: {', '.join(failed) if failed else 'None'}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print("=" * 60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Crisis Response AI Training Launcher')
    parser.add_argument('algorithm', choices=['dqn', 'reinforce', 'ppo', 'a2c', 'all'],
                       help='Algorithm to train')
    
    # Common arguments
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--visualize', action='store_true', help='Enable real-time visualization')
    parser.add_argument('--record', action='store_true', help='Record training video')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--variant', type=str, choices=['default', 'fast', 'stable', 'performance'],
                       default='default', help='Configuration variant')
    
    # Algorithm-specific arguments
    parser.add_argument('--trajectories', type=int, help='Number of trajectories (REINFORCE)')
    parser.add_argument('--steps', type=int, help='Rollout steps (PPO)')
    parser.add_argument('--epochs', type=int, help='Training epochs (PPO)')
    parser.add_argument('--n-steps', type=int, help='N-step returns (A2C)')
    parser.add_argument('--shared', action='store_true', help='Shared network (A2C)')
    parser.add_argument('--rms-prop', action='store_true', help='Use RMSprop (A2C)')
    
    args = parser.parse_args()
    
    print(f"🎯 Crisis Response AI Training Launcher")
    print(f"🤖 Algorithm: {args.algorithm.upper()}")
    print(f"📊 Episodes: {args.episodes}")
    print(f"⚙️  Variant: {args.variant}")
    
    if args.visualize:
        print("🎨 Visualization: Enabled")
    if args.record:
        print("📹 Video Recording: Enabled")
    if args.gpu:
        print("🚀 GPU Acceleration: Enabled")
    
    print("-" * 40)
    
    try:
        if args.algorithm == 'all':
            launch_all(args)
        elif args.algorithm == 'dqn':
            result = launch_dqn(args)
            sys.exit(result.returncode)
        elif args.algorithm == 'reinforce':
            result = launch_reinforce(args)
            sys.exit(result.returncode)
        elif args.algorithm == 'ppo':
            result = launch_ppo(args)
            sys.exit(result.returncode)
        elif args.algorithm == 'a2c':
            result = launch_a2c(args)
            sys.exit(result.returncode)
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()