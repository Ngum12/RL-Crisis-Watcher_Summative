#!/usr/bin/env python3
"""
REINFORCE Algorithm Training Script

Advanced REINFORCE training with comprehensive features:
- Policy gradient with baseline
- Generalized Advantage Estimation (GAE)
- Multiple trajectory collection
- Entropy regularization
- Automated hyperparameter optimization
- Real-time visualization and monitoring
"""

import sys
import os
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from environments.crisis_env import CrisisResponseEnv
from algorithms.reinforce import REINFORCEAgent, REINFORCEConfig
from visualization.real_time_renderer import CrisisRenderer
from visualization.charts import TrainingVisualizer
from visualization.recording import VideoRecorder

class REINFORCETrainer:
    """Professional REINFORCE training system"""
    
    def __init__(self, config_path: str = None, save_dir: str = "models/reinforce"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = REINFORCEConfig(**config_dict)
        else:
            self.config = self._get_optimized_config()
        
        # Initialize components
        self.environment = CrisisResponseEnv(
            num_regions=12,
            time_horizon=100,
            render_mode=None
        )
        
        self.agent = REINFORCEAgent(
            self.environment.observation_space,
            self.environment.action_space,
            self.config
        )
        
        self.visualizer = TrainingVisualizer()
        self.renderer = None
        self.video_recorder = None
        
        print("🎯 REINFORCE Training System Initialized")
        print(f"📊 Environment: {self.environment.observation_space.shape} → {self.environment.action_space.n}")
        print(f"🧠 Policy Network Parameters: {sum(p.numel() for p in self.agent.policy_network.parameters()):,}")
        if self.config.use_baseline:
            print(f"📈 Value Network Parameters: {sum(p.numel() for p in self.agent.value_network.parameters()):,}")
    
    def _get_optimized_config(self) -> REINFORCEConfig:
        """Get optimized REINFORCE configuration"""
        return REINFORCEConfig(
            # Optimized hyperparameters
            learning_rate=3e-4,
            baseline_lr=1e-3,
            gamma=0.99,
            
            # Policy gradient specific
            n_trajectories=10,
            use_baseline=True,
            normalize_advantages=True,
            advantage_method='gae',
            gae_lambda=0.95,
            
            # Network architecture
            hidden_layers=[256, 128, 64],
            activation='tanh',
            dropout_rate=0.1,
            
            # Regularization
            entropy_coef=0.01,
            gradient_clipping=5.0,
            weight_decay=1e-4,
            
            # Training parameters
            max_episodes=1000,
            max_trajectory_length=100,
            
            # Paths
            model_save_path=str(self.save_dir),
            log_save_path="logs/reinforce"
        )
    
    def train(self, episodes: int = None, enable_visualization: bool = False, 
              record_video: bool = False) -> dict:
        """Train REINFORCE agent with comprehensive monitoring"""
        
        if episodes is None:
            episodes = self.config.max_episodes
        
        print(f"🚀 Starting REINFORCE Training for {episodes} episodes")
        print("=" * 60)
        
        # Setup visualization
        if enable_visualization:
            self.renderer = CrisisRenderer(1400, 900)
        
        # Setup video recording
        if record_video:
            video_path = self.save_dir / "reinforce_training.mp4"
            self.video_recorder = VideoRecorder(str(video_path), fps=30)
            self.video_recorder.start_recording(1400, 900)
            print(f"📹 Recording training video: {video_path}")
        
        # Training loop with enhanced monitoring
        start_time = time.time()
        best_score = float('-inf')
        episode_rewards = []
        episode_lengths = []
        training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'advantages': []
        }
        
        try:
            for episode in range(episodes):
                episode_start = time.time()
                state = self.environment.reset()
                episode_reward = 0.0
                episode_length = 0
                
                # Reset trajectory for new episode
                self.agent.current_trajectory = self.agent.Trajectory() if hasattr(self.agent, 'Trajectory') else None
                
                for step in range(self.config.max_trajectory_length):
                    # Select action
                    action = self.agent.select_action(state, training=True)
                    
                    # Environment step
                    next_state, reward, done, info = self.environment.step(action)
                    
                    # Store experience and update
                    experience = (state, action, reward, next_state, done)
                    update_info = self.agent._process_experience(experience)
                    
                    # Record update information
                    if update_info:
                        if 'policy_loss' in update_info:
                            training_stats['policy_losses'].append(update_info['policy_loss'])
                        if 'value_loss' in update_info:
                            training_stats['value_losses'].append(update_info['value_loss'])
                        if 'entropy' in update_info:
                            training_stats['entropies'].append(update_info['entropy'])
                        if 'average_advantage' in update_info:
                            training_stats['advantages'].append(update_info['average_advantage'])
                    
                    # Update counters
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                    
                    # Render if enabled
                    if self.renderer:
                        self.renderer.render_complete_frame(self.environment)
                        if self.video_recorder:
                            self.video_recorder.add_pygame_frame(self.renderer.screen)
                    
                    if done:
                        break
                
                # Record episode statistics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                training_stats['episode_rewards'].append(episode_reward)
                training_stats['episode_lengths'].append(episode_length)
                
                # Log training progress
                if episode % self.config.log_frequency == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    avg_length = np.mean(episode_lengths[-10:])
                    episode_time = time.time() - episode_start
                    
                    # Get training statistics
                    training_stats_summary = self.agent.get_training_stats()
                    
                    print(f"Episode {episode:4d} | "
                          f"Reward: {episode_reward:7.2f} | "
                          f"Avg(10): {avg_reward:7.2f} | "
                          f"Length: {episode_length:3d} | "
                          f"Trajectories: {training_stats_summary.get('trajectories_collected', 0)} | "
                          f"Time: {episode_time:.2f}s")
                    
                    # Print loss information if available
                    if training_stats['policy_losses']:
                        recent_policy_loss = np.mean(training_stats['policy_losses'][-5:])
                        print(f"         Policy Loss: {recent_policy_loss:.4f} | "
                              f"Entropy: {training_stats_summary.get('avg_entropy', 0):.4f}")
                
                # Save best model
                if episode_reward > best_score:
                    best_score = episode_reward
                    best_model_path = self.save_dir / "best_model.pth"
                    self.agent.save_model(str(best_model_path))
                    print(f"🏆 New best score: {best_score:.2f} - Model saved!")
                
                # Periodic evaluation
                if episode % self.config.evaluation_frequency == 0 and episode > 0:
                    eval_score = self._evaluate_agent()
                    print(f"📊 Evaluation Score: {eval_score:.2f}")
                
                # Periodic model saving
                if episode % self.config.save_frequency == 0 and episode > 0:
                    checkpoint_path = self.save_dir / f"checkpoint_episode_{episode}.pth"
                    self.agent.save_model(str(checkpoint_path))
                
                # Log to visualizer
                self.visualizer.log_episode('REINFORCE', episode, episode_reward, episode_length)
                if training_stats['policy_losses']:
                    self.visualizer.log_step('REINFORCE', episode, loss=training_stats['policy_losses'][-1])
        
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
        
        finally:
            # Training completed
            training_time = time.time() - start_time
            
            # Save final model
            final_model_path = self.save_dir / "final_model.pth"
            self.agent.save_model(str(final_model_path))
            
            # Stop recording
            if self.video_recorder:
                self.video_recorder.stop_recording()
            
            # Cleanup renderer
            if self.renderer:
                self.renderer.cleanup()
            
            # Save training statistics
            stats_path = self.save_dir / "training_stats.json"
            with open(stats_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_stats = {}
                for key, value in training_stats.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], (np.ndarray, np.number)):
                            serializable_stats[key] = [float(v) if np.isscalar(v) else v.tolist() for v in value]
                        else:
                            serializable_stats[key] = value
                    else:
                        serializable_stats[key] = value
                
                json.dump(serializable_stats, f, indent=2)
            
            # Print summary
            print("\n" + "=" * 60)
            print("🎉 REINFORCE TRAINING COMPLETED!")
            print(f"⏱️  Training time: {training_time:.2f} seconds")
            print(f"🏆 Best score: {best_score:.2f}")
            print(f"📊 Average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"🎯 Episodes completed: {len(episode_rewards)}")
            print(f"💾 Models saved in: {self.save_dir}")
            print("=" * 60)
            
            return {
                'training_time': training_time,
                'best_score': best_score,
                'final_average': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
                'episodes_completed': len(episode_rewards),
                'training_stats': training_stats
            }
    
    def _evaluate_agent(self, num_episodes: int = 5) -> float:
        """Evaluate agent performance"""
        eval_rewards = []
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            episode_reward = 0.0
            
            for step in range(self.config.max_trajectory_length):
                action = self.agent.select_action(state, training=False)
                state, reward, done, _ = self.environment.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)
    
    def load_and_evaluate(self, model_path: str):
        """Load trained model and evaluate"""
        print(f"📂 Loading model from: {model_path}")
        self.agent.load_model(model_path)
        
        eval_score = self._evaluate_agent(num_episodes=10)
        print(f"📊 Evaluation score: {eval_score:.2f}")
        
        return eval_score


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train REINFORCE Agent for Crisis Response')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--save-dir', type=str, default='models/reinforce', help='Model save directory')
    parser.add_argument('--visualize', action='store_true', help='Enable real-time visualization')
    parser.add_argument('--record', action='store_true', help='Record training video')
    parser.add_argument('--evaluate', type=str, help='Path to model for evaluation only')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--trajectories', type=int, default=10, help='Number of trajectories per update')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
        print("🚀 Using GPU acceleration")
    else:
        device = 'cpu'
        print("💻 Using CPU")
    
    try:
        # Create trainer
        trainer = REINFORCETrainer(config_path=args.config, save_dir=args.save_dir)
        
        # Override number of trajectories if specified
        if args.trajectories:
            trainer.config.n_trajectories = args.trajectories
        
        if args.evaluate:
            # Evaluation mode
            trainer.load_and_evaluate(args.evaluate)
        else:
            # Training mode
            trainer.train(
                episodes=args.episodes,
                enable_visualization=args.visualize,
                record_video=args.record
            )
    
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()