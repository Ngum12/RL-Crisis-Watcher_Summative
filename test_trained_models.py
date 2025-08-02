#!/usr/bin/env python3
"""
Test Trained Models - Load and Evaluate Real RL Agents
Test the best saved models and demonstrate their performance
"""

import os
import glob
import numpy as np
from datetime import datetime

try:
    from stable_baselines3 import DQN, PPO, A2C
    from real_rl_training import CrisisResponseEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

class ModelTester:
    """Test and demonstrate trained RL models"""
    
    def __init__(self):
        self.models_dir = "models"
        self.results_dir = "results"
    
    def find_best_models(self):
        """Find all best model files"""
        best_models = {}
        
        for algorithm in ['dqn', 'ppo', 'a2c']:
            best_path = os.path.join(self.models_dir, f"{algorithm}_best", "best_model.zip")
            if os.path.exists(best_path):
                best_models[algorithm.upper()] = best_path
            else:
                print(f"⚠️  Best {algorithm.upper()} model not found at {best_path}")
        
        return best_models
    
    def load_model(self, algorithm: str, model_path: str):
        """Load a trained model"""
        try:
            if algorithm == "DQN":
                return DQN.load(model_path)
            elif algorithm == "PPO":
                return PPO.load(model_path)
            elif algorithm == "A2C":
                return A2C.load(model_path)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        except Exception as e:
            print(f"❌ Failed to load {algorithm} model: {e}")
            return None
    
    def test_model(self, model, algorithm: str, n_episodes: int = 5):
        """Test a model on the environment"""
        env = CrisisResponseEnv()
        
        print(f"\n🧪 Testing {algorithm} model ({n_episodes} episodes)...")
        
        episode_rewards = []
        total_lives_saved = 0
        total_crises_prevented = 0
        total_steps = 0
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            print(f"   Episode {episode + 1}: ", end="", flush=True)
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                total_lives_saved += info.get('lives_saved', 0)
                total_crises_prevented += info.get('crises_prevented', 0)
            
            episode_rewards.append(episode_reward)
            total_steps += episode_steps
            
            print(f"Reward: {episode_reward:.1f}, Steps: {episode_steps}, Lives: {info.get('total_lives_saved', 0)}")
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        print(f"📊 {algorithm} Results:")
        print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   Best Episode: {np.max(episode_rewards):.1f}")
        print(f"   Total Lives Saved: {total_lives_saved}")
        print(f"   Total Crises Prevented: {total_crises_prevented}")
        print(f"   Average Steps: {total_steps / n_episodes:.1f}")
        
        return {
            'algorithm': algorithm,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'total_lives_saved': total_lives_saved,
            'total_crises_prevented': total_crises_prevented,
            'avg_steps': total_steps / n_episodes
        }
    
    def demonstrate_best_model(self):
        """Find and demonstrate the best performing model"""
        best_models = self.find_best_models()
        
        if not best_models:
            print("❌ No trained models found!")
            print("Run real_rl_training.py first to train models.")
            return
        
        print("🏆 TESTING BEST TRAINED MODELS")
        print("=" * 50)
        
        all_results = []
        
        for algorithm, model_path in best_models.items():
            print(f"\n📦 Loading {algorithm} model from: {model_path}")
            
            model = self.load_model(algorithm, model_path)
            if model is None:
                continue
            
            results = self.test_model(model, algorithm)
            all_results.append(results)
        
        # Find overall best model
        if all_results:
            best_model = max(all_results, key=lambda x: x['mean_reward'])
            
            print(f"\n🥇 BEST PERFORMING MODEL: {best_model['algorithm']}")
            print(f"   Mean Reward: {best_model['mean_reward']:.2f}")
            print(f"   Lives Saved: {best_model['total_lives_saved']}")
            print(f"   Crises Prevented: {best_model['total_crises_prevented']}")
            
            return best_model
        
        return None
    
    def run_interactive_demo(self):
        """Run an interactive demonstration"""
        best_models = self.find_best_models()
        
        if not best_models:
            print("❌ No trained models found!")
            return
        
        print("\n🎮 INTERACTIVE CRISIS RESPONSE DEMO")
        print("=" * 40)
        print("Available models:")
        
        algorithms = list(best_models.keys())
        for i, alg in enumerate(algorithms):
            print(f"  {i+1}. {alg}")
        
        try:
            choice = int(input(f"\nChoose model (1-{len(algorithms)}): ")) - 1
            
            if 0 <= choice < len(algorithms):
                algorithm = algorithms[choice]
                model_path = best_models[algorithm]
                
                print(f"\n🚀 Loading {algorithm} model...")
                model = self.load_model(algorithm, model_path)
                
                if model is None:
                    return
                
                print(f"✅ {algorithm} model loaded successfully!")
                print("\n🌍 Starting crisis response simulation...")
                print("Watch the AI make decisions to manage crises in:")
                print("   • Cameroon (Anglophone tensions)")
                print("   • DR Congo (M23 rebels, mining conflicts)")  
                print("   • Sudan (SAF vs RSF conflict)")
                print()
                
                # Run interactive episode
                env = CrisisResponseEnv()
                obs, _ = env.reset()
                
                step = 0
                total_reward = 0
                
                print("🎯 Episode starting...")
                
                while step < 20:  # Show first 20 steps
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    
                    step += 1
                    total_reward += reward
                    
                    # Action descriptions
                    actions = [
                        "Monitor Situation", "Deploy Peacekeepers", "Economic Aid",
                        "Diplomatic Intervention", "Humanitarian Aid", "Early Warning",
                        "Media Campaign", "Intelligence Gathering"
                    ]
                    
                    print(f"Step {step:2d}: {actions[action]:20} | Reward: {reward:6.1f} | Target: {info.get('target_country', 'Unknown')}")
                    
                    if info.get('lives_saved', 0) > 0:
                        print(f"        💝 Lives saved: {info['lives_saved']}")
                    
                    if info.get('crises_prevented', 0) > 0:
                        print(f"        🛡️  Crisis prevented!")
                    
                    if done:
                        print(f"\n🏁 Episode completed at step {step}")
                        break
                
                print(f"\n📊 Episode Summary:")
                print(f"   Total Reward: {total_reward:.1f}")
                print(f"   Lives Saved: {info.get('total_lives_saved', 0)}")
                print(f"   Crises Prevented: {info.get('total_crises_prevented', 0)}")
                print(f"   Final Stability: {info.get('avg_stability', 0):.2%}")
                
        except (ValueError, KeyboardInterrupt):
            print("\nDemo cancelled.")

def main():
    """Main testing function"""
    if not SB3_AVAILABLE:
        print("❌ Stable Baselines3 not available!")
        print("Install with: pip install stable-baselines3")
        return
    
    tester = ModelTester()
    
    print("🧪 TRAINED MODEL TESTING SYSTEM")
    print("=" * 40)
    print("This script loads and tests your trained RL models")
    print()
    
    while True:
        print("Choose an option:")
        print("1. Test all best models")
        print("2. Interactive demo with specific model")  
        print("3. List available models")
        print("0. Exit")
        print()
        
        try:
            choice = input("Enter choice (0-3): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                tester.demonstrate_best_model()
            elif choice == "2":
                tester.run_interactive_demo()
            elif choice == "3":
                best_models = tester.find_best_models()
                print(f"\n📦 Available Models:")
                for alg, path in best_models.items():
                    print(f"   {alg}: {path}")
                if not best_models:
                    print("   No models found. Train some first!")
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            break
        
        print("\n" + "="*40 + "\n")
    
    print("👋 Goodbye!")
    input("Press ENTER to exit...")

if __name__ == "__main__":
    main()