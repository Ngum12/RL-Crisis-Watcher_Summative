"""
Training Utilities for Crisis Response AI

Common utilities and helper functions for training RL algorithms:
- Training configuration management
- Performance monitoring and logging
- Model checkpointing and saving
- Evaluation and testing utilities
- Training session management
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

@dataclass
class TrainingConfig:
    """Universal training configuration"""
    # Basic parameters
    algorithm: str = "DQN"
    episodes: int = 1000
    max_steps_per_episode: int = 100
    
    # Environment settings
    num_regions: int = 12
    time_horizon: int = 100
    
    # Training control
    save_frequency: int = 100
    evaluation_frequency: int = 50
    log_frequency: int = 10
    
    # Visualization and recording
    enable_visualization: bool = False
    record_video: bool = False
    
    # Paths
    save_dir: str = "models"
    log_dir: str = "logs"
    
    # Performance tracking
    target_reward: float = 200.0
    patience: int = 100  # Early stopping patience
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

class TrainingManager:
    """Manages training sessions with comprehensive monitoring"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Setup directories
        self.save_dir = Path(config.save_dir) / config.algorithm.lower()
        self.log_dir = Path(config.log_dir) / config.algorithm.lower()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.evaluation_scores = deque(maxlen=100)
        self.training_losses = deque(maxlen=10000)
        
        # Performance tracking
        self.best_score = float('-inf')
        self.best_model_path = None
        self.training_start_time = None
        self.last_improvement_episode = 0
        
        # Session metadata
        self.session_id = f"{config.algorithm}_{int(time.time())}"
        self.training_metadata = {
            'session_id': self.session_id,
            'algorithm': config.algorithm,
            'start_time': None,
            'end_time': None,
            'total_episodes': 0,
            'best_score': float('-inf'),
            'training_time': 0.0
        }
        
        self.logger.info(f"Training Manager initialized for {config.algorithm}")
        self.logger.info(f"Session ID: {self.session_id}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.log_dir / f"training_{self.config.algorithm}_{int(time.time())}.log"
        
        # Create logger
        self.logger = logging.getLogger(f"trainer_{self.config.algorithm}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def start_training(self):
        """Start training session"""
        self.training_start_time = time.time()
        self.training_metadata['start_time'] = self.training_start_time
        
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING TRAINING SESSION: {self.session_id}")
        self.logger.info(f"Algorithm: {self.config.algorithm}")
        self.logger.info(f"Target Episodes: {self.config.episodes}")
        self.logger.info(f"Target Reward: {self.config.target_reward}")
        self.logger.info("=" * 60)
    
    def log_episode(self, episode: int, reward: float, length: int, 
                   additional_info: Dict[str, Any] = None):
        """Log episode results"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Check for new best score
        if reward > self.best_score:
            self.best_score = reward
            self.last_improvement_episode = episode
            self.training_metadata['best_score'] = self.best_score
            
            self.logger.info(f"🏆 NEW BEST SCORE: {reward:.2f} at episode {episode}")
        
        # Log progress at specified frequency
        if episode % self.config.log_frequency == 0:
            avg_reward = np.mean(list(self.episode_rewards)[-10:])
            avg_length = np.mean(list(self.episode_lengths)[-10:])
            
            log_msg = (f"Episode {episode:4d} | "
                      f"Reward: {reward:7.2f} | "
                      f"Avg(10): {avg_reward:7.2f} | "
                      f"Length: {length:3d} | "
                      f"Best: {self.best_score:7.2f}")
            
            if additional_info:
                for key, value in additional_info.items():
                    if isinstance(value, float):
                        log_msg += f" | {key}: {value:.4f}"
                    else:
                        log_msg += f" | {key}: {value}"
            
            self.logger.info(log_msg)
    
    def log_training_loss(self, loss: float):
        """Log training loss"""
        self.training_losses.append(loss)
    
    def log_evaluation(self, episode: int, eval_score: float):
        """Log evaluation results"""
        self.evaluation_scores.append(eval_score)
        self.logger.info(f"📊 Evaluation at episode {episode}: {eval_score:.2f}")
    
    def should_stop_early(self, episode: int) -> bool:
        """Check if training should stop early"""
        if episode - self.last_improvement_episode > self.config.patience:
            self.logger.info(f"Early stopping: No improvement for {self.config.patience} episodes")
            return True
        
        # Check if target reward is reached
        if len(self.episode_rewards) >= 10:
            recent_avg = np.mean(list(self.episode_rewards)[-10:])
            if recent_avg >= self.config.target_reward:
                self.logger.info(f"Target reward {self.config.target_reward} reached!")
                return True
        
        return False
    
    def save_checkpoint(self, agent, episode: int, additional_data: Dict[str, Any] = None):
        """Save training checkpoint"""
        checkpoint_path = self.save_dir / f"checkpoint_episode_{episode}.pth"
        
        # Save agent model
        agent.save_model(str(checkpoint_path))
        
        # Save training state
        training_state = {
            'episode': episode,
            'best_score': self.best_score,
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'evaluation_scores': list(self.evaluation_scores),
            'training_metadata': self.training_metadata,
            'config': self.config.to_dict()
        }
        
        if additional_data:
            training_state.update(additional_data)
        
        state_path = self.save_dir / f"training_state_episode_{episode}.json"
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2, default=str)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_best_model(self, agent):
        """Save best performing model"""
        best_path = self.save_dir / "best_model.pth"
        agent.save_model(str(best_path))
        self.best_model_path = str(best_path)
        self.logger.info(f"Best model saved: {best_path}")
    
    def finalize_training(self, total_episodes: int):
        """Finalize training session"""
        training_time = time.time() - self.training_start_time
        
        self.training_metadata.update({
            'end_time': time.time(),
            'total_episodes': total_episodes,
            'training_time': training_time
        })
        
        # Save final training summary
        summary = self.generate_training_summary()
        summary_path = self.save_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Log final results
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED!")
        self.logger.info(f"Total Episodes: {total_episodes}")
        self.logger.info(f"Best Score: {self.best_score:.2f}")
        self.logger.info(f"Training Time: {training_time:.2f} seconds")
        self.logger.info(f"Final Average (last 100): {np.mean(list(self.episode_rewards)[-100:]):.2f}")
        self.logger.info(f"Summary saved: {summary_path}")
        self.logger.info("=" * 60)
        
        return summary
    
    def generate_training_summary(self) -> Dict[str, Any]:
        """Generate comprehensive training summary"""
        rewards = list(self.episode_rewards)
        lengths = list(self.episode_lengths)
        
        summary = {
            'session_metadata': self.training_metadata,
            'performance_metrics': {
                'total_episodes': len(rewards),
                'best_score': self.best_score,
                'mean_reward': float(np.mean(rewards)) if rewards else 0,
                'std_reward': float(np.std(rewards)) if rewards else 0,
                'final_100_mean': float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards)) if rewards else 0,
                'mean_episode_length': float(np.mean(lengths)) if lengths else 0,
                'convergence_episode': self.last_improvement_episode
            },
            'training_curves': {
                'episode_rewards': rewards,
                'episode_lengths': lengths,
                'evaluation_scores': list(self.evaluation_scores),
                'training_losses': list(self.training_losses)
            },
            'config': self.config.to_dict(),
            'paths': {
                'best_model': self.best_model_path,
                'save_directory': str(self.save_dir),
                'log_directory': str(self.log_dir)
            }
        }
        
        return summary
    
    def create_performance_plots(self):
        """Create performance visualization plots"""
        if not self.episode_rewards:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.config.algorithm} Training Performance', fontsize=16)
        
        episodes = range(len(self.episode_rewards))
        
        # Episode rewards
        ax1.plot(episodes, self.episode_rewards, alpha=0.7, linewidth=1)
        if len(self.episode_rewards) > 20:
            # Moving average
            moving_avg = np.convolve(self.episode_rewards, np.ones(20)/20, mode='valid')
            ax1.plot(range(19, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Average (20)')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Episode lengths
        ax2.plot(episodes, self.episode_lengths, alpha=0.7, color='green')
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True, alpha=0.3)
        
        # Training losses
        if self.training_losses:
            ax3.plot(self.training_losses, alpha=0.7, color='orange')
            ax3.set_title('Training Loss')
            ax3.set_xlabel('Update Step')
            ax3.set_ylabel('Loss')
            ax3.grid(True, alpha=0.3)
        
        # Evaluation scores
        if self.evaluation_scores:
            eval_episodes = range(0, len(self.evaluation_scores) * self.config.evaluation_frequency, 
                                self.config.evaluation_frequency)
            ax4.plot(eval_episodes, self.evaluation_scores, 'o-', color='purple')
            ax4.set_title('Evaluation Scores')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Evaluation Score')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / f"{self.config.algorithm}_training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance plots saved: {plot_path}")

def load_training_state(state_path: str) -> Dict[str, Any]:
    """Load training state from file"""
    with open(state_path, 'r') as f:
        return json.load(f)

def compare_training_sessions(session_paths: List[str]) -> Dict[str, Any]:
    """Compare multiple training sessions"""
    sessions = {}
    
    for path in session_paths:
        try:
            session_data = load_training_state(path)
            algorithm = session_data['config']['algorithm']
            sessions[algorithm] = session_data
        except Exception as e:
            print(f"Error loading session {path}: {e}")
    
    # Generate comparison
    comparison = {
        'algorithms': list(sessions.keys()),
        'performance_comparison': {},
        'best_performer': None
    }
    
    best_score = float('-inf')
    best_algorithm = None
    
    for algorithm, data in sessions.items():
        metrics = data['performance_metrics']
        comparison['performance_comparison'][algorithm] = metrics
        
        if metrics['best_score'] > best_score:
            best_score = metrics['best_score']
            best_algorithm = algorithm
    
    comparison['best_performer'] = best_algorithm
    
    return comparison