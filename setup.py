#!/usr/bin/env python3
"""
Crisis Response AI - Setup Script

Automated setup for the Crisis Response AI reinforcement learning project.
This script handles:
- Dependency installation
- Environment verification
- Directory structure creation
- Initial configuration
- System compatibility checks
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'models', 'logs', 'reports', 'demos', 'results', 
        'training', 'evaluation', 'data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("✅ Directory structure created")

def check_gpu_availability():
    """Check for GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU detected - training will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check GPU")
        return False

def verify_environment():
    """Verify environment is properly set up"""
    print("🔍 Verifying environment setup...")
    
    try:
        # Test imports
        import gym
        import torch
        import numpy
        import matplotlib
        import pygame
        import optuna
        
        print("✅ All core dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def create_sample_config():
    """Create sample configuration file"""
    config = {
        "environment": {
            "num_regions": 12,
            "time_horizon": 100,
            "render_mode": "human"
        },
        "training": {
            "max_episodes": 1000,
            "parallel_training": True,
            "save_frequency": 100,
            "evaluation_frequency": 50,
            "visualization_enabled": True,
            "video_recording": True
        },
        "hyperparameter_optimization": {
            "enabled": True,
            "n_trials": 50,
            "optimization_timeout": 1800,
            "study_name": "crisis_response_optimization"
        },
        "algorithms": {
            "DQN": {"enabled": True, "optimize_hyperparameters": True},
            "REINFORCE": {"enabled": True, "optimize_hyperparameters": True},
            "PPO": {"enabled": True, "optimize_hyperparameters": True},
            "A2C": {"enabled": True, "optimize_hyperparameters": True}
        }
    }
    
    import json
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Sample configuration created: config.json")

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("🎉 CRISIS RESPONSE AI - SETUP COMPLETE!")
    print("="*60)
    print("\n🚀 Quick Start Commands:")
    print("  1. Demo Environment:     python demo_environment.py")
    print("  2. Train All Models:     python train_all_models.py")
    print("  3. Generate Report:      python generate_report.py results/training_results.json")
    
    print("\n📊 Advanced Usage:")
    print("  • Train specific algorithm:  python train_all_models.py --algorithms DQN")
    print("  • Skip optimization:         python train_all_models.py --no-optimization")
    print("  • Record demo video:         python demo_environment.py --record")
    print("  • Custom config:             python train_all_models.py --config config.json")
    
    print("\n📁 Project Structure:")
    print("  • environments/    - Custom RL environments")
    print("  • algorithms/      - RL algorithm implementations") 
    print("  • visualization/   - Advanced rendering and charts")
    print("  • evaluation/      - Performance analysis tools")
    print("  • models/          - Saved trained models")
    print("  • reports/         - Generated analysis reports")
    print("  • demos/           - Video demonstrations")
    
    print("\n💡 Tips for Success:")
    print("  • Start with demo_environment.py to understand the problem")
    print("  • Use smaller episode counts for quick testing")
    print("  • Check logs/ directory for detailed training information")
    print("  • GPU recommended for faster training")
    
    print("\n🆘 Need Help?")
    print("  • Check README.md for detailed documentation")
    print("  • Review sample config.json for configuration options")
    print("  • See reports/ for example analysis outputs")
    
    print("\n🏆 Ready to revolutionize crisis response with AI!")
    print("="*60)

def main():
    """Main setup function"""
    print("🌍 Crisis Response AI - Automated Setup")
    print("Building a groundbreaking RL system for conflict prediction!")
    print("="*60)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    print(f"💻 Platform: {platform.system()} {platform.release()}")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Continuing with existing dependencies...")
    
    # Check GPU
    check_gpu_availability()
    
    # Verify environment
    if not verify_environment():
        print("⚠️  Some dependencies may be missing - check manually")
    
    # Create sample config
    create_sample_config()
    
    # Print instructions
    print_usage_instructions()
    
    print("\n✅ Setup completed successfully!")

if __name__ == "__main__":
    main()