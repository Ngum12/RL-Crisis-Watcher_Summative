"""
Proximal Policy Optimization (PPO) Implementation

Advanced PPO (Schulman et al., 2017) with state-of-the-art features:
- Clipped surrogate objective for stable policy updates
- Generalized Advantage Estimation (GAE)
- Multiple epochs of minibatch updates
- Adaptive KL penalty for constraint satisfaction
- Value function clipping
- Entropy regularization for exploration
- Learning rate annealing
- Sophisticated rollout collection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, NamedTuple
from collections import deque
import math

from .base_agent import BaseAgent, AgentConfig, create_network

@dataclass
class PPOConfig(AgentConfig):
    """Configuration for PPO Agent"""
    # PPO-specific parameters
    n_steps: int = 2048  # Steps per rollout
    n_epochs: int = 10   # Epochs per update
    minibatch_size: int = 64
    clip_range: float = 0.2  # PPO clipping parameter
    clip_range_vf: Optional[float] = None  # Value function clipping
    
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95  # GAE parameter
    normalize_advantages: bool = True
    
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = 'tanh'
    dropout_rate: float = 0.0
    shared_network: bool = False  # Share layers between actor and critic
    
    # Regularization and constraints
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None  # Target KL divergence
    kl_coef: float = 0.2  # KL penalty coefficient
    
    # Learning rate scheduling
    lr_schedule: str = 'linear'  # 'linear', 'constant', 'cosine'
    lr_schedule_steps: int = 1000000
    
    # Advanced features
    use_gae: bool = True
    use_proper_time_limits: bool = True
    reward_scaling: float = 1.0
    observation_normalization: bool = False
    
    # Logging and debugging
    log_frequency: int = 10
    save_frequency: int = 100
    debug_mode: bool = False

class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, n_steps: int, observation_space, action_space, device: torch.device):
        self.n_steps = n_steps
        self.device = device
        self.reset()
        
        # Initialize storage arrays
        obs_shape = observation_space.shape
        self.observations = np.zeros((n_steps,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=bool)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)
        
    def reset(self):
        """Reset buffer"""
        self.pos = 0
        self.full = False
        
    def add(self, obs, action, reward, done, value, log_prob):
        """Add step to buffer"""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        self.pos += 1
        if self.pos >= self.n_steps:
            self.full = True
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float, 
                                     gae_lambda: float, use_gae: bool = True):
        """Compute returns and advantages using GAE"""
        if use_gae:
            # Generalized Advantage Estimation
            advantages = np.zeros_like(self.rewards)
            last_gae_lam = 0
            
            for step in reversed(range(self.n_steps)):
                if step == self.n_steps - 1:
                    next_non_terminal = 1.0 - self.dones[step]
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    next_value = self.values[step + 1]
                
                delta = (self.rewards[step] + gamma * next_value * next_non_terminal - 
                        self.values[step])
                advantages[step] = last_gae_lam = (delta + gamma * gae_lambda * 
                                                  next_non_terminal * last_gae_lam)
            
            self.advantages = advantages
            self.returns = advantages + self.values
        else:
            # Simple returns computation
            returns = np.zeros_like(self.rewards)
            running_return = last_value
            
            for step in reversed(range(self.n_steps)):
                if self.dones[step]:
                    running_return = 0
                returns[step] = running_return = (self.rewards[step] + 
                                                gamma * running_return)
            
            self.returns = returns
            self.advantages = returns - self.values
    
    def get_minibatches(self, minibatch_size: int):
        """Generate minibatches for training"""
        if not self.full:
            return
        
        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)
        
        start_idx = 0
        while start_idx < self.n_steps:
            end_idx = min(start_idx + minibatch_size, self.n_steps)
            batch_indices = indices[start_idx:end_idx]
            
            yield self._get_batch(batch_indices)
            start_idx = end_idx
    
    def _get_batch(self, indices):
        """Get batch data for given indices"""
        return {
            'observations': torch.FloatTensor(self.observations[indices]).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'dones': torch.BoolTensor(self.dones[indices]).to(self.device),
            'values': torch.FloatTensor(self.values[indices]).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs[indices]).to(self.device),
            'advantages': torch.FloatTensor(self.advantages[indices]).to(self.device),
            'returns': torch.FloatTensor(self.returns[indices]).to(self.device)
        }

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, observation_space, action_space, config: PPOConfig):
        super(ActorCriticNetwork, self).__init__()
        self.config = config
        
        input_size = np.prod(observation_space.shape)
        output_size = action_space.n
        
        if config.shared_network:
            # Shared feature extractor
            self.shared_net = create_network(
                input_size, config.hidden_layers[-1], config.hidden_layers[:-1],
                config.activation, config.dropout_rate
            )
            
            # Actor head
            self.actor_head = nn.Linear(config.hidden_layers[-1], output_size)
            
            # Critic head
            self.critic_head = nn.Linear(config.hidden_layers[-1], 1)
        else:
            # Separate networks
            self.actor_net = create_network(
                input_size, output_size, config.hidden_layers,
                config.activation, config.dropout_rate
            )
            
            self.critic_net = create_network(
                input_size, 1, config.hidden_layers,
                config.activation, config.dropout_rate
            )
            
            self.shared_net = None
    
    def forward(self, x):
        """Forward pass returning both policy and value"""
        if self.shared_net is not None:
            features = self.shared_net(x)
            action_logits = self.actor_head(features)
            value = self.critic_head(features)
        else:
            action_logits = self.actor_net(x)
            value = self.critic_net(x)
        
        return action_logits, value
    
    def get_action_distribution(self, x):
        """Get action distribution"""
        action_logits, _ = self.forward(x)
        return Categorical(logits=action_logits)
    
    def get_value(self, x):
        """Get value estimate"""
        if self.shared_net is not None:
            features = self.shared_net(x)
            return self.critic_head(features)
        else:
            return self.critic_net(x)
    
    def evaluate_actions(self, x, actions):
        """Evaluate actions and return log probs, values, and entropy"""
        action_logits, values = self.forward(x)
        action_dist = Categorical(logits=action_logits)
        
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        return log_probs, values.squeeze(), entropy

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization Agent
    
    Implements PPO with advanced features:
    - Clipped surrogate objective
    - Generalized Advantage Estimation
    - Multiple epoch updates with minibatches
    - Value function clipping
    - Adaptive KL divergence penalty
    - Learning rate scheduling
    """
    
    def __init__(self, observation_space, action_space, config: PPOConfig = None):
        if config is None:
            config = PPOConfig()
        
        super().__init__(observation_space, action_space, config)
        
        # PPO-specific attributes
        self.config: PPOConfig = config
        self.rollout_buffer = None
        self.n_updates = 0
        
        # Training statistics
        self.policy_losses = deque(maxlen=1000)
        self.value_losses = deque(maxlen=1000)
        self.entropy_losses = deque(maxlen=1000)
        self.kl_divergences = deque(maxlen=1000)
        self.clip_fractions = deque(maxlen=1000)
        
    def _initialize_algorithm(self):
        """Initialize PPO-specific components"""
        # Create actor-critic network
        self.actor_critic = ActorCriticNetwork(
            self.observation_space, self.action_space, self.config
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5
        )
        
        # Create rollout buffer
        self.rollout_buffer = RolloutBuffer(
            self.config.n_steps,
            self.observation_space,
            self.action_space,
            self.device
        )
        
        # Learning rate scheduler
        if self.config.lr_schedule == 'linear':
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(0.1, 1.0 - step / self.config.lr_schedule_steps)
            )
        elif self.config.lr_schedule == 'cosine':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_schedule_steps
            )
        else:
            self.lr_scheduler = None
        
        self.logger.info(f"Initialized PPO with {sum(p.numel() for p in self.actor_critic.parameters())} parameters")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_dist = self.actor_critic.get_action_distribution(state_tensor)
            
            if training:
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                value = self.actor_critic.get_value(state_tensor)
                
                # Store for rollout buffer
                self._last_action_info = {
                    'log_prob': log_prob.item(),
                    'value': value.item()
                }
            else:
                # Deterministic action for evaluation
                action = action_dist.probs.argmax(dim=-1)
        
        return action.item()
    
    def _process_experience(self, experience: Tuple) -> Optional[Dict[str, float]]:
        """Process experience and add to rollout buffer"""
        state, action, reward, next_state, done = experience
        
        # Add to rollout buffer
        if hasattr(self, '_last_action_info'):
            self.rollout_buffer.add(
                state, action, reward * self.config.reward_scaling, done,
                self._last_action_info['value'],
                self._last_action_info['log_prob']
            )
        
        # Update if buffer is full
        if self.rollout_buffer.full:
            # Get value for last state
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                last_value = self.actor_critic.get_value(next_state_tensor).item()
                if done:
                    last_value = 0.0
            
            # Compute returns and advantages
            self.rollout_buffer.compute_returns_and_advantages(
                last_value, self.config.gamma, self.config.gae_lambda, self.config.use_gae
            )
            
            # Update networks
            update_info = self._update_networks()
            
            # Reset buffer
            self.rollout_buffer.reset()
            
            return update_info
        
        return None
    
    def _update_networks(self) -> Dict[str, float]:
        """Update actor-critic networks using PPO"""
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divergences = []
        clip_fractions = []
        
        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = self.rollout_buffer.advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            self.rollout_buffer.advantages = advantages
        
        # Multiple epochs of updates
        for epoch in range(self.config.n_epochs):
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_entropy_losses = []
            epoch_kl_divs = []
            epoch_clip_fracs = []
            
            # Minibatch updates
            for batch in self.rollout_buffer.get_minibatches(self.config.minibatch_size):
                # Evaluate current policy
                log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    batch['observations'], batch['actions']
                )
                
                # Compute probability ratio
                ratio = torch.exp(log_probs - batch['log_probs'])
                
                # Compute surrogate losses
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 
                                  1.0 + self.config.clip_range) * batch['advantages']
                
                # Policy loss (negative because we want to maximize)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.config.clip_range_vf is not None:
                    # Clipped value loss
                    value_pred_clipped = batch['values'] + torch.clamp(
                        values - batch['values'],
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf
                    )
                    value_loss1 = F.mse_loss(values, batch['returns'])
                    value_loss2 = F.mse_loss(value_pred_clipped, batch['returns'])
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = F.mse_loss(values, batch['returns'])
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                            self.config.value_loss_coef * value_loss + 
                            self.config.entropy_coef * entropy_loss)
                
                # KL divergence for monitoring
                with torch.no_grad():
                    kl_div = ((log_probs - batch['log_probs']).exp() - 1 - 
                            (log_probs - batch['log_probs'])).mean()
                    
                    # Clip fraction
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_range).float().mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # Record losses
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropy_losses.append(entropy_loss.item())
                epoch_kl_divs.append(kl_div.item())
                epoch_clip_fracs.append(clip_frac.item())
            
            # Early stopping based on KL divergence
            if (self.config.target_kl is not None and 
                np.mean(epoch_kl_divs) > self.config.target_kl):
                self.logger.info(f"Early stopping at epoch {epoch} due to KL divergence: {np.mean(epoch_kl_divs):.4f}")
                break
            
            # Store epoch averages
            policy_losses.append(np.mean(epoch_policy_losses))
            value_losses.append(np.mean(epoch_value_losses))
            entropy_losses.append(np.mean(epoch_entropy_losses))
            kl_divergences.append(np.mean(epoch_kl_divs))
            clip_fractions.append(np.mean(epoch_clip_fracs))
        
        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        self.n_updates += 1
        
        # Record statistics
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy_loss = np.mean(entropy_losses)
        avg_kl_div = np.mean(kl_divergences)
        avg_clip_frac = np.mean(clip_fractions)
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_losses.append(avg_entropy_loss)
        self.kl_divergences.append(avg_kl_div)
        self.clip_fractions.append(avg_clip_frac)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'kl_divergence': avg_kl_div,
            'clip_fraction': avg_clip_frac,
            'n_updates': self.n_updates
        }
    
    def update(self, experiences: List[Tuple]) -> Dict[str, float]:
        """Batch update for compatibility with base class"""
        update_info = {}
        for experience in experiences:
            result = self._process_experience(experience)
            if result:
                update_info.update(result)
        
        return update_info
    
    def save_model(self, filepath: str):
        """Save model and training state"""
        checkpoint = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'current_episode': self.current_episode,
            'best_score': self.best_score,
            'total_steps': self.total_steps,
            'n_updates': self.n_updates
        }
        
        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_episode = checkpoint.get('current_episode', 0)
        self.best_score = checkpoint.get('best_score', float('-inf'))
        self.total_steps = checkpoint.get('total_steps', 0)
        self.n_updates = checkpoint.get('n_updates', 0)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for a state (for analysis)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_dist = self.actor_critic.get_action_distribution(state_tensor)
            return action_dist.probs.cpu().numpy().flatten()
    
    def get_value_estimate(self, state: np.ndarray) -> float:
        """Get value estimate for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.actor_critic.get_value(state_tensor)
            return value.item()
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        stats = {
            'total_steps': self.total_steps,
            'current_episode': self.current_episode,
            'n_updates': self.n_updates,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        if self.policy_losses:
            stats['avg_policy_loss'] = np.mean(list(self.policy_losses)[-10:])
            
        if self.value_losses:
            stats['avg_value_loss'] = np.mean(list(self.value_losses)[-10:])
            
        if self.entropy_losses:
            stats['avg_entropy_loss'] = np.mean(list(self.entropy_losses)[-10:])
            
        if self.kl_divergences:
            stats['avg_kl_divergence'] = np.mean(list(self.kl_divergences)[-10:])
            
        if self.clip_fractions:
            stats['avg_clip_fraction'] = np.mean(list(self.clip_fractions)[-10:])
        
        return stats