"""
REINFORCE Algorithm Implementation

Advanced REINFORCE (Williams, 1992) with modern improvements:
- Baseline for variance reduction
- Multiple trajectory collection
- Gradient clipping for stability
- Entropy regularization for exploration
- Adaptive learning rates
- Sophisticated advantage estimation
- Trajectory importance sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import math

from .base_agent import BaseAgent, AgentConfig, create_network

@dataclass
class REINFORCEConfig(AgentConfig):
    """Configuration for REINFORCE Agent"""
    # REINFORCE-specific parameters
    use_baseline: bool = True
    baseline_lr: float = 1e-3
    n_trajectories: int = 10  # Number of trajectories to collect per update
    max_trajectory_length: int = 1000
    
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    normalize_advantages: bool = True
    
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = 'tanh'
    dropout_rate: float = 0.1
    
    # Regularization
    entropy_coef: float = 0.01  # Entropy regularization coefficient
    gradient_clipping: float = 5.0
    weight_decay: float = 1e-4
    
    # Advanced features
    advantage_method: str = 'returns'  # 'returns', 'gae', 'td'
    gae_lambda: float = 0.95  # For GAE advantage estimation
    trajectory_sampling: str = 'uniform'  # 'uniform', 'importance'
    
    # Learning rate scheduling
    lr_schedule: bool = True
    lr_schedule_step: int = 1000
    lr_schedule_gamma: float = 0.9
    
    # Performance tracking
    log_frequency: int = 10
    save_frequency: int = 100

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE"""
    
    def __init__(self, input_size: int, output_size: int, config: REINFORCEConfig):
        super(PolicyNetwork, self).__init__()
        self.config = config
        
        # Create policy network
        self.policy_net = create_network(
            input_size, output_size, config.hidden_layers,
            config.activation, config.dropout_rate
        )
        
        # Add final softmax for probability distribution
        self.output_layer = nn.Sequential(
            self.policy_net,
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action probabilities"""
        return self.output_layer(x)
    
    def get_action_distribution(self, x: torch.Tensor) -> Categorical:
        """Get action distribution for sampling"""
        probs = self.forward(x)
        # Add small epsilon to avoid numerical issues
        probs = probs + 1e-8
        return Categorical(probs)

class ValueNetwork(nn.Module):
    """Value network for baseline estimation"""
    
    def __init__(self, input_size: int, config: REINFORCEConfig):
        super(ValueNetwork, self).__init__()
        
        # Create value network
        self.value_net = create_network(
            input_size, 1, config.hidden_layers,
            config.activation, config.dropout_rate
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value"""
        return self.value_net(x)

class Trajectory:
    """Container for trajectory data"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []  # For baseline
        self.dones = []
        
    def add_step(self, state, action, reward, log_prob, value=None, done=False):
        """Add a step to the trajectory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)
        self.dones.append(done)
    
    def compute_returns(self, gamma: float, normalize: bool = True) -> torch.Tensor:
        """Compute discounted returns"""
        returns = []
        G = 0
        
        # Compute returns backwards
        for reward in reversed(self.rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        
        if normalize and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def compute_advantages_gae(self, values: torch.Tensor, gamma: float, 
                              lambda_: float, normalize: bool = True) -> torch.Tensor:
        """Compute advantages using Generalized Advantage Estimation"""
        advantages = []
        advantage = 0
        
        # Add a zero value for the terminal state
        values_ext = torch.cat([values, torch.zeros(1)])
        
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values_ext[t]
                advantage = delta
            else:
                delta = self.rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
                advantage = delta + gamma * lambda_ * advantage
            
            advantages.insert(0, advantage)
        
        advantages = torch.FloatTensor(advantages)
        
        if normalize and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def compute_td_advantages(self, values: torch.Tensor, gamma: float, 
                             normalize: bool = True) -> torch.Tensor:
        """Compute advantages using TD error"""
        advantages = []
        values_ext = torch.cat([values, torch.zeros(1)])
        
        for t in range(len(self.rewards)):
            if self.dones[t]:
                advantage = self.rewards[t] - values_ext[t]
            else:
                advantage = self.rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
            
            advantages.append(advantage)
        
        advantages = torch.FloatTensor(advantages)
        
        if normalize and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages

class REINFORCEAgent(BaseAgent):
    """
    Advanced REINFORCE Agent
    
    Implements the REINFORCE algorithm with multiple modern improvements:
    - Baseline for variance reduction
    - Multiple trajectory collection for stable gradients
    - Generalized Advantage Estimation (GAE)
    - Entropy regularization for exploration
    - Sophisticated gradient clipping and regularization
    """
    
    def __init__(self, observation_space, action_space, config: REINFORCEConfig = None):
        if config is None:
            config = REINFORCEConfig()
        
        super().__init__(observation_space, action_space, config)
        
        # REINFORCE-specific attributes
        self.config: REINFORCEConfig = config
        self.current_trajectory = None
        self.trajectories = []
        self.policy_losses = deque(maxlen=1000)
        self.value_losses = deque(maxlen=1000)
        self.entropies = deque(maxlen=1000)
        
    def _initialize_algorithm(self):
        """Initialize REINFORCE-specific components"""
        input_size = np.prod(self.observation_space.shape)
        output_size = self.action_space.n
        
        # Create policy network
        self.policy_network = PolicyNetwork(input_size, output_size, self.config).to(self.device)
        
        # Create value network for baseline
        if self.config.use_baseline:
            self.value_network = ValueNetwork(input_size, self.config).to(self.device)
            
            # Separate optimizers for policy and value networks
            self.policy_optimizer = optim.Adam(
                self.policy_network.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            self.value_optimizer = optim.Adam(
                self.value_network.parameters(),
                lr=self.config.baseline_lr,
                weight_decay=self.config.weight_decay
            )
            
            # Learning rate schedulers
            if self.config.lr_schedule:
                self.policy_scheduler = optim.lr_scheduler.StepLR(
                    self.policy_optimizer,
                    step_size=self.config.lr_schedule_step,
                    gamma=self.config.lr_schedule_gamma
                )
                
                self.value_scheduler = optim.lr_scheduler.StepLR(
                    self.value_optimizer,
                    step_size=self.config.lr_schedule_step,
                    gamma=self.config.lr_schedule_gamma
                )
        else:
            self.value_network = None
            self.policy_optimizer = optim.Adam(
                self.policy_network.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            if self.config.lr_schedule:
                self.policy_scheduler = optim.lr_scheduler.StepLR(
                    self.policy_optimizer,
                    step_size=self.config.lr_schedule_step,
                    gamma=self.config.lr_schedule_gamma
                )
        
        # Initialize trajectory collection
        self.current_trajectory = Trajectory()
        
        self.logger.info(f"Initialized REINFORCE with {sum(p.numel() for p in self.policy_network.parameters())} policy parameters")
        if self.config.use_baseline:
            self.logger.info(f"Value network has {sum(p.numel() for p in self.value_network.parameters())} parameters")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_dist = self.policy_network.get_action_distribution(state_tensor)
            
            if training:
                # Sample action from distribution
                action = action_dist.sample()
            else:
                # Use most probable action for evaluation
                action = action_dist.probs.argmax(dim=-1)
            
            action_item = action.item()
            log_prob = action_dist.log_prob(action).item()
            
            # Get value estimate if using baseline
            value = None
            if self.config.use_baseline and self.value_network is not None:
                value = self.value_network(state_tensor).item()
        
        # Store for trajectory if training
        if training and self.current_trajectory is not None:
            self.current_trajectory.log_probs.append(log_prob)
            if value is not None:
                self.current_trajectory.values.append(value)
        
        return action_item
    
    def _process_experience(self, experience: Tuple) -> Optional[Dict[str, float]]:
        """Process experience and add to current trajectory"""
        state, action, reward, next_state, done = experience
        
        if self.current_trajectory is not None:
            # The log_prob and value were already stored in select_action
            self.current_trajectory.states.append(state)
            self.current_trajectory.actions.append(action)
            self.current_trajectory.rewards.append(reward)
            self.current_trajectory.dones.append(done)
            
            if done:
                # Trajectory completed
                self.trajectories.append(self.current_trajectory)
                self.current_trajectory = Trajectory()
                
                # Update if we have enough trajectories
                if len(self.trajectories) >= self.config.n_trajectories:
                    update_info = self._update_networks()
                    self.trajectories = []  # Clear trajectories after update
                    return update_info
        
        return None
    
    def _update_networks(self) -> Dict[str, float]:
        """Update policy and value networks using collected trajectories"""
        if not self.trajectories:
            return {}
        
        policy_losses = []
        value_losses = []
        entropies = []
        
        for trajectory in self.trajectories:
            if len(trajectory.states) == 0:
                continue
            
            # Convert trajectory data to tensors
            states = torch.FloatTensor(trajectory.states).to(self.device)
            actions = torch.LongTensor(trajectory.actions).to(self.device)
            log_probs = torch.FloatTensor(trajectory.log_probs).to(self.device)
            
            # Compute advantages based on chosen method
            if self.config.advantage_method == 'returns':
                returns = trajectory.compute_returns(self.config.gamma, self.config.normalize_advantages)
                returns = returns.to(self.device)
                
                if self.config.use_baseline and self.value_network is not None:
                    values = torch.FloatTensor(trajectory.values).to(self.device)
                    advantages = returns - values
                else:
                    advantages = returns
                    
            elif self.config.advantage_method == 'gae' and self.config.use_baseline:
                values = torch.FloatTensor(trajectory.values).to(self.device)
                advantages = trajectory.compute_advantages_gae(
                    values, self.config.gamma, self.config.gae_lambda,
                    self.config.normalize_advantages
                ).to(self.device)
                returns = advantages + values
                
            elif self.config.advantage_method == 'td' and self.config.use_baseline:
                values = torch.FloatTensor(trajectory.values).to(self.device)
                advantages = trajectory.compute_td_advantages(
                    values, self.config.gamma, self.config.normalize_advantages
                ).to(self.device)
                returns = advantages + values
            else:
                # Fallback to simple returns
                returns = trajectory.compute_returns(self.config.gamma, self.config.normalize_advantages)
                returns = returns.to(self.device)
                advantages = returns
            
            # Compute policy loss
            policy_loss = -(log_probs * advantages.detach()).mean()
            
            # Compute entropy for regularization
            action_dist = self.policy_network.get_action_distribution(states)
            entropy = action_dist.entropy().mean()
            
            # Total policy loss with entropy regularization
            total_policy_loss = policy_loss - self.config.entropy_coef * entropy
            
            policy_losses.append(total_policy_loss)
            entropies.append(entropy.item())
            
            # Compute value loss if using baseline
            if self.config.use_baseline and self.value_network is not None:
                predicted_values = self.value_network(states).squeeze()
                value_loss = F.mse_loss(predicted_values, returns.detach())
                value_losses.append(value_loss)
        
        # Average losses across trajectories
        avg_policy_loss = torch.stack(policy_losses).mean()
        avg_entropy = np.mean(entropies)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        avg_policy_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(),
                self.config.gradient_clipping
            )
        
        self.policy_optimizer.step()
        
        # Update value network if using baseline
        avg_value_loss = 0.0
        if self.config.use_baseline and value_losses:
            avg_value_loss = torch.stack(value_losses).mean()
            
            self.value_optimizer.zero_grad()
            avg_value_loss.backward()
            
            if self.config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.value_network.parameters(),
                    self.config.gradient_clipping
                )
            
            self.value_optimizer.step()
            avg_value_loss = avg_value_loss.item()
        
        # Update learning rate schedulers
        if self.config.lr_schedule:
            self.policy_scheduler.step()
            if self.config.use_baseline:
                self.value_scheduler.step()
        
        # Record losses
        policy_loss_item = avg_policy_loss.item()
        self.policy_losses.append(policy_loss_item)
        if avg_value_loss > 0:
            self.value_losses.append(avg_value_loss)
        self.entropies.append(avg_entropy)
        
        return {
            'policy_loss': policy_loss_item,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'n_trajectories': len(self.trajectories)
        }
    
    def update(self, experiences: List[Tuple]) -> Dict[str, float]:
        """Batch update for compatibility with base class"""
        # REINFORCE collects full trajectories, so this method
        # processes experiences sequentially
        update_info = {}
        for experience in experiences:
            result = self._process_experience(experience)
            if result:
                update_info.update(result)
        
        return update_info
    
    def _train_episode(self, environment) -> Tuple[float, int, Optional[float]]:
        """Override to handle trajectory-based training"""
        state = environment.reset()
        episode_reward = 0.0
        episode_length = 0
        
        # Start new trajectory
        self.current_trajectory = Trajectory()
        
        for step in range(self.config.max_steps_per_episode):
            # Select action
            action = self.select_action(state, training=True)
            
            # Take step in environment
            next_state, reward, done, info = environment.step(action)
            
            # Process experience
            experience = (state, action, reward, next_state, done)
            update_info = self._process_experience(experience)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            if done:
                break
        
        # If episode ended but we haven't updated yet, force an update
        if len(self.trajectories) > 0:
            update_info = self._update_networks()
            self.trajectories = []
        
        avg_loss = np.mean(list(self.policy_losses)[-10:]) if self.policy_losses else None
        return episode_reward, episode_length, avg_loss
    
    def save_model(self, filepath: str):
        """Save model and training state"""
        checkpoint = {
            'policy_network_state_dict': self.policy_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'config': self.config.to_dict(),
            'current_episode': self.current_episode,
            'best_score': self.best_score,
            'total_steps': self.total_steps
        }
        
        if self.config.use_baseline and self.value_network is not None:
            checkpoint['value_network_state_dict'] = self.value_network.state_dict()
            checkpoint['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
        
        if self.config.lr_schedule:
            checkpoint['policy_scheduler_state_dict'] = self.policy_scheduler.state_dict()
            if self.config.use_baseline:
                checkpoint['value_scheduler_state_dict'] = self.value_scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        
        if self.config.use_baseline and 'value_network_state_dict' in checkpoint:
            self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        if self.config.lr_schedule and 'policy_scheduler_state_dict' in checkpoint:
            self.policy_scheduler.load_state_dict(checkpoint['policy_scheduler_state_dict'])
            if self.config.use_baseline and 'value_scheduler_state_dict' in checkpoint:
                self.value_scheduler.load_state_dict(checkpoint['value_scheduler_state_dict'])
        
        self.current_episode = checkpoint.get('current_episode', 0)
        self.best_score = checkpoint.get('best_score', float('-inf'))
        self.total_steps = checkpoint.get('total_steps', 0)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_policy_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for a state (for analysis)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs = self.policy_network(state_tensor)
            return probs.cpu().numpy().flatten()
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        stats = {
            'total_steps': self.total_steps,
            'current_episode': self.current_episode,
            'trajectories_collected': len(self.trajectories)
        }
        
        if self.policy_losses:
            stats['avg_policy_loss'] = np.mean(list(self.policy_losses)[-10:])
            
        if self.value_losses:
            stats['avg_value_loss'] = np.mean(list(self.value_losses)[-10:])
            
        if self.entropies:
            stats['avg_entropy'] = np.mean(list(self.entropies)[-10:])
        
        return stats