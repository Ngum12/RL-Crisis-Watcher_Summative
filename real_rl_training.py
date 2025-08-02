#!/usr/bin/env python3
"""
Real RL Training System - Actual Neural Networks
Train real DQN, PPO, REINFORCE, A2C agents on Crisis Response Environment

This script trains actual RL models using Stable Baselines3 and saves:
- Best models in models/
- Training logs in logs/
- Results and plots in results/
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

# RL Libraries
try:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# Crisis Environment
class CrisisResponseEnv(gym.Env):
    """
    Real Crisis Response Environment for RL Training
    Cameroon, DRC, Sudan Crisis Management
    """
    
    def __init__(self):
        super().__init__()
        
        # Environment parameters
        self.n_countries = 3  # Cameroon, DRC, Sudan
        self.n_features_per_country = 12  # Various crisis indicators
        
        # Action space: 8 different actions (Monitor, Deploy PK, Aid, etc.)
        self.action_space = gym.spaces.Discrete(8)
        
        # Observation space: 36-dimensional (12 features * 3 countries)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.n_countries * self.n_features_per_country,), 
            dtype=np.float32
        )
        
        # Country names for logging
        self.countries = ["Cameroon", "DR_Congo", "Sudan"]
        
        # Initialize state
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize realistic crisis state for each country
        # Cameroon: Moderate crisis (Anglophone tensions)
        cameroon_state = np.array([
            0.4, 0.5, 0.3, 0.4,  # Political, Economic, Social, Security
            0.7, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0  # Crisis indicators
        ], dtype=np.float32)
        
        # DR Congo: Severe crisis (M23, mining conflicts)
        drc_state = np.array([
            0.2, 0.3, 0.3, 0.2,  # Political, Economic, Social, Security
            0.9, 0.8, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0  # Crisis indicators
        ], dtype=np.float32)
        
        # Sudan: Emergency (SAF vs RSF)
        sudan_state = np.array([
            0.1, 0.2, 0.2, 0.1,  # Political, Economic, Social, Security
            0.95, 0.9, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0  # Crisis indicators
        ], dtype=np.float32)
        
        # Combine all country states
        self.state = np.concatenate([cameroon_state, drc_state, sudan_state])
        
        # Reset episode variables
        self.step_count = 0
        self.max_steps = 200
        self.total_reward = 0
        self.lives_saved = 0
        self.crises_prevented = 0
        
        return self.state, {}
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        action = int(action)
        
        # Determine target country (focus on highest crisis level)
        country_crisis_levels = []
        for i in range(self.n_countries):
            start_idx = i * self.n_features_per_country
            crisis_indicators = self.state[start_idx + 4:start_idx + 8]  # Crisis type indicators
            crisis_level = np.sum(crisis_indicators)
            country_crisis_levels.append(crisis_level)
        
        target_country = np.argmax(country_crisis_levels)
        country_start = target_country * self.n_features_per_country
        
        # Execute action
        reward = 0
        lives_saved_this_step = 0
        crises_prevented_this_step = 0
        
        if action == 0:  # Monitor
            reward = 1
            
        elif action == 1:  # Deploy Peacekeepers
            # Improve security situation
            self.state[country_start + 3] = min(1.0, self.state[country_start + 3] + 0.15)
            # Reduce armed conflict crisis
            self.state[country_start + 4] = max(0.0, self.state[country_start + 4] - 0.1)
            reward = 25
            lives_saved_this_step = np.random.randint(50, 200)
            
        elif action == 2:  # Economic Aid
            # Improve economic situation
            self.state[country_start + 1] = min(1.0, self.state[country_start + 1] + 0.2)
            # Reduce resource conflict
            self.state[country_start + 5] = max(0.0, self.state[country_start + 5] - 0.15)
            reward = 20
            lives_saved_this_step = np.random.randint(30, 150)
            
        elif action == 3:  # Diplomatic Intervention
            # Improve political situation
            self.state[country_start] = min(1.0, self.state[country_start] + 0.25)
            # Reduce political crisis
            self.state[country_start + 6] = max(0.0, self.state[country_start + 6] - 0.2)
            reward = 30
            lives_saved_this_step = np.random.randint(20, 100)
            
        elif action == 4:  # Humanitarian Aid
            # Improve social cohesion
            self.state[country_start + 2] = min(1.0, self.state[country_start + 2] + 0.1)
            # Reduce refugee crisis
            self.state[country_start + 7] = max(0.0, self.state[country_start + 7] - 0.1)
            reward = 15
            lives_saved_this_step = np.random.randint(100, 300)
            
        elif action == 5:  # Early Warning
            # Prevent crisis escalation
            for crisis_idx in range(4, 8):
                self.state[country_start + crisis_idx] *= 0.95
            reward = 10
            crises_prevented_this_step = 1
            
        elif action == 6:  # Media Campaign
            # Improve social cohesion
            self.state[country_start + 2] = min(1.0, self.state[country_start + 2] + 0.05)
            reward = 8
            
        elif action == 7:  # Intelligence Gathering
            # Slight improvement to all stability factors
            for i in range(4):
                self.state[country_start + i] = min(1.0, self.state[country_start + i] + 0.02)
            reward = 5
        
        # Natural deterioration and random events
        self._update_environment()
        
        # Calculate bonus rewards
        stability_bonus = np.mean(self.state[0:4]) + np.mean(self.state[12:16]) + np.mean(self.state[24:28])
        reward += stability_bonus * 5
        
        # Update counters
        self.step_count += 1
        self.total_reward += reward
        self.lives_saved += lives_saved_this_step
        self.crises_prevented += crises_prevented_this_step
        
        # Episode termination
        done = self.step_count >= self.max_steps
        
        # Additional termination conditions
        avg_stability = np.mean([
            np.mean(self.state[0:4]),    # Cameroon
            np.mean(self.state[12:16]),  # DR Congo  
            np.mean(self.state[24:28])   # Sudan
        ])
        
        if avg_stability < 0.1:  # Complete collapse
            done = True
            reward -= 100
        elif avg_stability > 0.9:  # Peace achieved
            done = True
            reward += 100
        
        # Info dictionary
        info = {
            'lives_saved': lives_saved_this_step,
            'crises_prevented': crises_prevented_this_step,
            'total_lives_saved': self.lives_saved,
            'total_crises_prevented': self.crises_prevented,
            'target_country': self.countries[target_country],
            'avg_stability': avg_stability,
            'episode_reward': self.total_reward
        }
        
        return self.state.copy(), reward, done, False, info
    
    def _update_environment(self):
        """Update environment with natural changes and random events"""
        # Natural fluctuations
        for i in range(len(self.state)):
            # Small random changes
            self.state[i] += np.random.normal(0, 0.01)
            self.state[i] = np.clip(self.state[i], 0.0, 1.0)
        
        # Random crisis events (2% chance)
        if np.random.random() < 0.02:
            country = np.random.randint(self.n_countries)
            crisis_type = np.random.randint(4, 8)  # Crisis indicators
            country_start = country * self.n_features_per_country
            
            # Escalate crisis
            self.state[country_start + crisis_type] = min(1.0, self.state[country_start + crisis_type] + 0.2)
            
            # Worsen stability
            for i in range(4):
                self.state[country_start + i] *= 0.9

class RealRLTrainer:
    """Real RL Training System using Stable Baselines3"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.models_dir = "models"
        self.logs_dir = "logs"
        self.results_dir = "results"
        
        for dir_name in [self.models_dir, self.logs_dir, self.results_dir]:
            os.makedirs(dir_name, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Training results
        self.training_results = {}
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.logs_dir, f"training_{self.timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def train_dqn(self, total_timesteps: int = 50000) -> Dict[str, Any]:
        """Train DQN agent"""
        self.logger.info("🚀 Starting DQN training...")
        
        # Create environment
        env = CrisisResponseEnv()
        env = Monitor(env, os.path.join(self.logs_dir, f"dqn_monitor_{self.timestamp}"))
        
        # Create model
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=0.0001,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            max_grad_norm=10,
            tensorboard_log=None,  # Disabled to avoid dependency issues
            verbose=1
        )
        
        # Training callbacks
        eval_env = CrisisResponseEnv()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.models_dir, "dqn_best"),
            log_path=os.path.join(self.logs_dir, "dqn_eval"),
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(self.models_dir, "dqn_checkpoints"),
            name_prefix="dqn_model"
        )
        
        # Train the model
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            log_interval=100
        )
        training_time = time.time() - start_time
        
        # Save final model
        model.save(os.path.join(self.models_dir, f"dqn_final_{self.timestamp}"))
        
        # Evaluate trained model
        results = self._evaluate_model(model, "DQN", training_time)
        
        self.logger.info(f"✅ DQN training completed! Final reward: {results['mean_reward']:.2f}")
        return results
    
    def train_ppo(self, total_timesteps: int = 50000) -> Dict[str, Any]:
        """Train PPO agent"""
        self.logger.info("🚀 Starting PPO training...")
        
        # Create environment
        env = make_vec_env(lambda: CrisisResponseEnv(), n_envs=1)
        
        # Create model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=None,  # Disabled to avoid dependency issues
            verbose=1
        )
        
        # Training callbacks
        eval_env = CrisisResponseEnv()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.models_dir, "ppo_best"),
            log_path=os.path.join(self.logs_dir, "ppo_eval"),
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(self.models_dir, "ppo_checkpoints"),
            name_prefix="ppo_model"
        )
        
        # Train the model
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            log_interval=100
        )
        training_time = time.time() - start_time
        
        # Save final model
        model.save(os.path.join(self.models_dir, f"ppo_final_{self.timestamp}"))
        
        # Evaluate trained model
        results = self._evaluate_model(model, "PPO", training_time)
        
        self.logger.info(f"✅ PPO training completed! Final reward: {results['mean_reward']:.2f}")
        return results
    
    def train_a2c(self, total_timesteps: int = 50000) -> Dict[str, Any]:
        """Train A2C agent"""
        self.logger.info("🚀 Starting A2C training...")
        
        # Create environment
        env = make_vec_env(lambda: CrisisResponseEnv(), n_envs=1)
        
        # Create model
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.25,
            max_grad_norm=0.5,
            rms_prop_eps=1e-05,
            use_rms_prop=True,
            tensorboard_log=None,  # Disabled to avoid dependency issues
            verbose=1
        )
        
        # Training callbacks
        eval_env = CrisisResponseEnv()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.models_dir, "a2c_best"),
            log_path=os.path.join(self.logs_dir, "a2c_eval"),
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(self.models_dir, "a2c_checkpoints"),
            name_prefix="a2c_model"
        )
        
        # Train the model
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            log_interval=100
        )
        training_time = time.time() - start_time
        
        # Save final model
        model.save(os.path.join(self.models_dir, f"a2c_final_{self.timestamp}"))
        
        # Evaluate trained model
        results = self._evaluate_model(model, "A2C", training_time)
        
        self.logger.info(f"✅ A2C training completed! Final reward: {results['mean_reward']:.2f}")
        return results
    
    def _evaluate_model(self, model, algorithm_name: str, training_time: float, n_eval_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate trained model"""
        env = CrisisResponseEnv()
        
        episode_rewards = []
        total_lives_saved = 0
        total_crises_prevented = 0
        
        for episode in range(n_eval_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                
                total_lives_saved += info.get('lives_saved', 0)
                total_crises_prevented += info.get('crises_prevented', 0)
            
            episode_rewards.append(episode_reward)
        
        results = {
            'algorithm': algorithm_name,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'total_lives_saved': total_lives_saved,
            'total_crises_prevented': total_crises_prevented,
            'training_time': training_time,
            'n_eval_episodes': n_eval_episodes
        }
        
        return results
    
    def generate_comparison_report(self):
        """Generate comprehensive training comparison report"""
        if not self.training_results:
            self.logger.warning("No training results to compare")
            return
        
        # Create comparison plots
        self._plot_training_comparison()
        
        # Generate detailed report
        report_path = os.path.join(self.results_dir, f"real_rl_training_report_{self.timestamp}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_report_content())
        
        # Save results as JSON
        results_path = os.path.join(self.results_dir, f"real_rl_results_{self.timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        self.logger.info(f"📊 Report saved to: {report_path}")
        self.logger.info(f"💾 Results saved to: {results_path}")
    
    def _plot_training_comparison(self):
        """Create training comparison plots"""
        if not self.training_results:
            return
        
        algorithms = list(self.training_results.keys())
        mean_rewards = [self.training_results[alg]['mean_reward'] for alg in algorithms]
        lives_saved = [self.training_results[alg]['total_lives_saved'] for alg in algorithms]
        crises_prevented = [self.training_results[alg]['total_crises_prevented'] for alg in algorithms]
        training_times = [self.training_results[alg]['training_time'] for alg in algorithms]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mean rewards
        bars1 = ax1.bar(algorithms, mean_rewards, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Mean Episode Rewards')
        ax1.set_ylabel('Reward')
        for bar, value in zip(bars1, mean_rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Lives saved
        bars2 = ax2.bar(algorithms, lives_saved, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('Total Lives Saved (Evaluation)')
        ax2.set_ylabel('Lives')
        for bar, value in zip(bars2, lives_saved):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{value}', ha='center', va='bottom')
        
        # Crises prevented
        bars3 = ax3.bar(algorithms, crises_prevented, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_title('Crises Prevented (Evaluation)')
        ax3.set_ylabel('Count')
        for bar, value in zip(bars3, crises_prevented):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value}', ha='center', va='bottom')
        
        # Training time
        bars4 = ax4.bar(algorithms, training_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax4.set_title('Training Time')
        ax4.set_ylabel('Seconds')
        for bar, value in zip(bars4, training_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.0f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, f"real_rl_comparison_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📈 Comparison plots saved to: {plot_path}")
    
    def _generate_report_content(self) -> str:
        """Generate detailed report content"""
        best_algorithm = max(self.training_results.items(), key=lambda x: x[1]['mean_reward'])
        
        report = f"""# Real RL Training Results - Africa Crisis Response

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Environment**: Crisis Response (Cameroon, DR Congo, Sudan)
**Algorithms**: {", ".join(self.training_results.keys())}
**Training Library**: Stable Baselines3

---

## Executive Summary

Trained real neural network agents on African crisis response scenarios.

**Best Performing Algorithm**: {best_algorithm[0]} (Mean Reward: {best_algorithm[1]['mean_reward']:.2f})

---

## Detailed Results

| Algorithm | Mean Reward | Std Reward | Lives Saved | Crises Prevented | Training Time (s) |
|-----------|-------------|------------|-------------|------------------|-------------------|
"""
        
        for alg, results in self.training_results.items():
            report += f"| {alg} | {results['mean_reward']:.2f} | {results['std_reward']:.2f} | {results['total_lives_saved']} | {results['total_crises_prevented']} | {results['training_time']:.0f} |\n"
        
        report += f"""
---

## Algorithm Analysis

"""
        
        for alg, results in self.training_results.items():
            report += f"""
### {alg}

- **Mean Reward**: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}
- **Performance Range**: {results['min_reward']:.2f} to {results['max_reward']:.2f}
- **Lives Saved**: {results['total_lives_saved']} (evaluation episodes)
- **Crises Prevented**: {results['total_crises_prevented']}
- **Training Time**: {results['training_time']:.0f} seconds

"""
        
        report += f"""
---

## Model Files

**Saved Models:**
- Best models: `models/{{algorithm}}_best/best_model.zip`
- Final models: `models/{{algorithm}}_final_{self.timestamp}.zip`
- Checkpoints: `models/{{algorithm}}_checkpoints/`

**Logs:**
- Training logs: `logs/training_{self.timestamp}.log`
- Evaluation logs: `logs/{{algorithm}}_eval/`
- Monitor logs: `logs/{{algorithm}}_monitor_*/`

---

## Usage

Load trained models:
```python
from stable_baselines3 import DQN, PPO, A2C

# Load best model
model = DQN.load("models/dqn_best/best_model")

# Evaluate on environment
env = CrisisResponseEnv()
obs, _ = env.reset()
action, _ = model.predict(obs)
```

---

*Report generated by Real RL Training System*
"""
        
        return report

def main():
    """Main training function"""
    if not SB3_AVAILABLE:
        print("❌ Stable Baselines3 not available!")
        print("Install with: pip install stable-baselines3")
        return
    
    print("🌍 REAL RL TRAINING - AFRICA CRISIS RESPONSE")
    print("=" * 60)
    print("🧠 Training actual neural network agents")
    print("💾 Saving models, logs, and results")
    print("🎯 Environment: Cameroon, DR Congo, Sudan")
    print()
    
    trainer = RealRLTrainer()
    
    # Training parameters
    timesteps = 25000  # Reduced for faster training, increase for better performance
    
    try:
        # Train all algorithms
        algorithms_to_train = [
            ("DQN", trainer.train_dqn),
            ("PPO", trainer.train_ppo),
            ("A2C", trainer.train_a2c)
        ]
        
        for alg_name, train_func in algorithms_to_train:
            print(f"\n🚀 Training {alg_name}...")
            print(f"⏱️  Timesteps: {timesteps:,}")
            
            results = train_func(timesteps)
            trainer.training_results[alg_name] = results
            
            print(f"✅ {alg_name} completed:")
            print(f"   Mean Reward: {results['mean_reward']:.2f}")
            print(f"   Lives Saved: {results['total_lives_saved']}")
            print(f"   Training Time: {results['training_time']:.1f}s")
        
        # Generate comparison report
        print(f"\n📊 Generating comparison report...")
        trainer.generate_comparison_report()
        
        print(f"\n🏆 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Generated Files:")
        print(f"   • Models: models/ directory")
        print(f"   • Logs: logs/ directory")
        print(f"   • Results: results/ directory")
        print()
        print("🎯 Best models are automatically saved!")
        print("📈 Training logs saved to logs/ directory")
        print("💡 Install tensorboard for detailed training curves:")
        print("   pip install tensorboard")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        trainer.logger.error(f"Training failed: {e}")
    
    print()
    input("Press ENTER to exit...")

if __name__ == "__main__":
    main()