"""
Base Agent Class for Reinforcement Learning Algorithms

Provides common functionality and interfaces for all RL agents including:
- Standardized training and evaluation interfaces
- Performance monitoring and logging
- Model saving and loading
- Hyperparameter management
- Experience collection and replay
- Advanced exploration strategies
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import os
from collections import deque
import logging
import time
from pathlib import Path

@dataclass
class AgentConfig:
    """Base configuration for RL agents"""
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = 'relu'
    dropout_rate: float = 0.1
    
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Training control
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    update_frequency: int = 4
    target_update_frequency: int = 100
    
    # Performance tracking
    log_frequency: int = 10
    save_frequency: int = 100
    evaluation_frequency: int = 50
    evaluation_episodes: int = 5
    
    # Memory and storage
    memory_size: int = 100000
    min_memory_size: int = 1000
    model_save_path: str = 'models/'
    log_save_path: str = 'logs/'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class BaseAgent(ABC):
    """
    Abstract base class for all RL agents
    
    Provides standardized interfaces and common functionality:
    - Training and evaluation loops
    - Performance monitoring
    - Model persistence
    - Hyperparameter management
    - Experience replay (if applicable)
    """
    
    def __init__(self, observation_space, action_space, config: AgentConfig):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.training_losses = deque(maxlen=10000)
        self.evaluation_scores = deque(maxlen=100)
        
        # Training state
        self.current_episode = 0
        self.total_steps = 0
        self.training_start_time = None
        self.best_score = float('-inf')
        
        # Device setup
        self.device = torch.device(config.device)
        
        # Logging setup
        self.setup_logging()
        
        # Create save directories
        Path(config.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(config.log_save_path).mkdir(parents=True, exist_ok=True)
        
        # Algorithm-specific initialization
        self._initialize_algorithm()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_filename = f"{self.config.log_save_path}/{self.__class__.__name__}_{int(time.time())}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__} with config: {self.config.to_dict()}")
    
    @abstractmethod
    def _initialize_algorithm(self):
        """Initialize algorithm-specific components"""
        pass
    
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """Select action given current state"""
        pass
    
    @abstractmethod
    def update(self, experiences: List[Tuple]) -> Dict[str, float]:
        """Update agent based on experiences"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """Save model to file"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """Load model from file"""
        pass
    
    def train(self, environment, num_episodes: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Main training loop
        
        Args:
            environment: Gym environment
            num_episodes: Number of episodes to train (defaults to config)
            
        Returns:
            Training statistics
        """
        if num_episodes is None:
            num_episodes = self.config.max_episodes
        
        self.training_start_time = time.time()
        training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'training_losses': [],
            'evaluation_scores': []
        }
        
        self.logger.info(f"Starting training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            self.current_episode = episode
            episode_reward, episode_length, episode_loss = self._train_episode(environment)
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            training_stats['episode_rewards'].append(episode_reward)
            training_stats['episode_lengths'].append(episode_length)
            
            if episode_loss is not None:
                self.training_losses.append(episode_loss)
                training_stats['training_losses'].append(episode_loss)
            
            # Logging
            if episode % self.config.log_frequency == 0:
                self._log_training_progress(episode, episode_reward, episode_length, episode_loss)
            
            # Evaluation
            if episode % self.config.evaluation_frequency == 0:
                eval_score = self.evaluate(environment, self.config.evaluation_episodes)
                self.evaluation_scores.append(eval_score)
                training_stats['evaluation_scores'].append(eval_score)
                
                # Save best model
                if eval_score > self.best_score:
                    self.best_score = eval_score
                    self.save_model(f"{self.config.model_save_path}/best_model.pth")
                    self.logger.info(f"New best score: {eval_score:.2f} - Model saved")
            
            # Periodic model saving
            if episode % self.config.save_frequency == 0:
                self.save_model(f"{self.config.model_save_path}/checkpoint_episode_{episode}.pth")
        
        training_time = time.time() - self.training_start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model and statistics
        self.save_model(f"{self.config.model_save_path}/final_model.pth")
        self._save_training_statistics(training_stats)
        
        return training_stats
    
    def _train_episode(self, environment) -> Tuple[float, int, Optional[float]]:
        """Train for one episode"""
        state = environment.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_losses = []
        
        for step in range(self.config.max_steps_per_episode):
            # Select action
            action = self.select_action(state, training=True)
            
            # Take step in environment
            next_state, reward, done, info = environment.step(action)
            
            # Store experience and update
            experience = (state, action, reward, next_state, done)
            update_info = self._process_experience(experience)
            
            if update_info and 'loss' in update_info:
                episode_losses.append(update_info['loss'])
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            if done:
                break
        
        avg_loss = np.mean(episode_losses) if episode_losses else None
        return episode_reward, episode_length, avg_loss
    
    def _process_experience(self, experience: Tuple) -> Optional[Dict[str, float]]:
        """Process single experience (to be overridden by specific algorithms)"""
        return None
    
    def evaluate(self, environment, num_episodes: int = 5) -> float:
        """Evaluate agent performance"""
        total_rewards = []
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0.0
            
            for step in range(self.config.max_steps_per_episode):
                action = self.select_action(state, training=False)
                state, reward, done, _ = environment.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        self.logger.info(f"Evaluation: {avg_reward:.2f} ± {std_reward:.2f} over {num_episodes} episodes")
        return avg_reward
    
    def _log_training_progress(self, episode: int, reward: float, length: int, loss: Optional[float]):
        """Log training progress"""
        avg_reward = np.mean(list(self.episode_rewards)[-10:]) if len(self.episode_rewards) >= 10 else reward
        avg_length = np.mean(list(self.episode_lengths)[-10:]) if len(self.episode_lengths) >= 10 else length
        
        log_msg = f"Episode {episode}: Reward={reward:.2f}, Length={length}, "
        log_msg += f"Avg Reward (10)={avg_reward:.2f}, Avg Length (10)={avg_length:.1f}"
        
        if loss is not None:
            log_msg += f", Loss={loss:.4f}"
        
        self.logger.info(log_msg)
    
    def _save_training_statistics(self, stats: Dict[str, List[float]]):
        """Save training statistics to file"""
        stats_file = f"{self.config.log_save_path}/training_stats_{self.__class__.__name__}_{int(time.time())}.json"
        
        # Convert to serializable format
        serializable_stats = {}
        for key, values in stats.items():
            serializable_stats[key] = [float(v) for v in values]
        
        # Add metadata
        serializable_stats['metadata'] = {
            'algorithm': self.__class__.__name__,
            'total_episodes': len(stats['episode_rewards']),
            'total_steps': self.total_steps,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0,
            'best_score': self.best_score,
            'config': self.config.to_dict()
        }
        
        with open(stats_file, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        self.logger.info(f"Training statistics saved to {stats_file}")
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of agent performance"""
        if not self.episode_rewards:
            return {}
        
        rewards = list(self.episode_rewards)
        lengths = list(self.episode_lengths)
        
        summary = {
            'total_episodes': len(rewards),
            'total_steps': self.total_steps,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'mean_length': np.mean(lengths),
            'best_score': self.best_score
        }
        
        # Recent performance (last 100 episodes)
        if len(rewards) >= 100:
            recent_rewards = rewards[-100:]
            summary['recent_mean_reward'] = np.mean(recent_rewards)
            summary['recent_std_reward'] = np.std(recent_rewards)
        
        # Learning progress (improvement over time)
        if len(rewards) >= 50:
            first_half = rewards[:len(rewards)//2]
            second_half = rewards[len(rewards)//2:]
            summary['improvement'] = np.mean(second_half) - np.mean(first_half)
        
        return summary
    
    def set_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about action space"""
        if hasattr(self.action_space, 'n'):
            return {
                'type': 'discrete',
                'size': self.action_space.n,
                'shape': None
            }
        else:
            return {
                'type': 'continuous',
                'size': self.action_space.shape[0],
                'shape': self.action_space.shape,
                'low': self.action_space.low.tolist(),
                'high': self.action_space.high.tolist()
            }
    
    def get_state_space_info(self) -> Dict[str, Any]:
        """Get information about observation space"""
        return {
            'shape': self.observation_space.shape,
            'size': np.prod(self.observation_space.shape),
            'low': self.observation_space.low.tolist() if hasattr(self.observation_space, 'low') else None,
            'high': self.observation_space.high.tolist() if hasattr(self.observation_space, 'high') else None
        }

class Memory:
    """Experience replay memory for DQN and other algorithms"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, experience: Tuple):
        """Add experience to memory"""
        self.memory.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch from memory"""
        return list(np.random.choice(self.memory, size=batch_size, replace=False))
    
    def can_sample(self, batch_size: int) -> bool:
        """Check if we can sample a batch"""
        return len(self.memory) >= batch_size
    
    def __len__(self):
        return len(self.memory)

def create_network(input_size: int, output_size: int, hidden_layers: List[int], 
                  activation: str = 'relu', dropout_rate: float = 0.1) -> nn.Module:
    """Create neural network with specified architecture"""
    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_size, hidden_layers[0]))
    
    # Hidden layers
    for i in range(len(hidden_layers)):
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        
        # Dropout
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Next layer
        if i < len(hidden_layers) - 1:
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
    
    # Output layer
    layers.append(nn.Linear(hidden_layers[-1], output_size))
    
    return nn.Sequential(*layers)