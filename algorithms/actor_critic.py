"""
Actor-Critic (A2C) Algorithm Implementation

Advanced Actor-Critic with modern improvements:
- Synchronous advantage actor-critic
- N-step returns for reduced variance
- Generalized Advantage Estimation (GAE)
- Entropy regularization for exploration
- Value function bootstrapping
- Shared and separate network architectures
- Advanced exploration strategies
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

from .base_agent import BaseAgent, AgentConfig, create_network

@dataclass
class A2CConfig(AgentConfig):
    """Configuration for A2C Agent"""
    # A2C-specific parameters
    n_steps: int = 5  # Number of steps for n-step returns
    use_gae: bool = True
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    
    # Learning parameters
    learning_rate: float = 7e-4
    gamma: float = 0.99
    
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = 'relu'
    dropout_rate: float = 0.1
    shared_network: bool = True  # Share parameters between actor and critic
    
    # Regularization
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    weight_decay: float = 1e-5
    
    # Learning rate scheduling
    lr_schedule: bool = True
    lr_schedule_step: int = 10000
    lr_schedule_gamma: float = 0.95
    
    # Advanced features
    use_rms_prop: bool = True  # Use RMSprop instead of Adam
    alpha: float = 0.99  # RMSprop alpha
    eps: float = 1e-5  # RMSprop epsilon
    
    # Exploration
    exploration_method: str = 'entropy'  # 'entropy', 'epsilon', 'noise'
    epsilon_start: float = 0.1
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Performance tracking
    log_frequency: int = 10
    save_frequency: int = 100

class A2CNetwork(nn.Module):
    """Actor-Critic Network"""
    
    def __init__(self, observation_space, action_space, config: A2CConfig):
        super(A2CNetwork, self).__init__()
        self.config = config
        
        input_size = np.prod(observation_space.shape)
        output_size = action_space.n
        
        if config.shared_network:
            # Shared feature extractor
            self.shared_layers = create_network(
                input_size, config.hidden_layers[-1], config.hidden_layers[:-1],
                config.activation, config.dropout_rate
            )
            
            # Actor head (policy)
            self.actor_head = nn.Linear(config.hidden_layers[-1], output_size)
            
            # Critic head (value function)
            self.critic_head = nn.Linear(config.hidden_layers[-1], 1)
            
            # Initialize weights
            self._init_weights()
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
            
            self.shared_layers = None
            self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass returning action logits and value"""
        if self.shared_layers is not None:
            shared_features = self.shared_layers(x)
            action_logits = self.actor_head(shared_features)
            value = self.critic_head(shared_features)
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
        if self.shared_layers is not None:
            shared_features = self.shared_layers(x)
            return self.critic_head(shared_features)
        else:
            return self.critic_net(x)
    
    def evaluate_actions(self, x, actions):
        """Evaluate actions and return log probabilities, values, and entropy"""
        action_logits, values = self.forward(x)
        action_dist = Categorical(logits=action_logits)
        
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        return log_probs, values.squeeze(), entropy

class ExperienceBuffer:
    """Buffer for storing n-step experiences"""
    
    def __init__(self, n_steps: int, gamma: float):
        self.n_steps = n_steps
        self.gamma = gamma
        self.reset()
    
    def reset(self):
        """Reset buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done):
        """Add experience to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, next_value: float, use_gae: bool = True, 
                                     gae_lambda: float = 0.95, normalize: bool = True):
        """Compute returns and advantages"""
        if len(self.states) == 0:
            return [], []
        
        values = self.values + [next_value]
        
        if use_gae:
            # Generalized Advantage Estimation
            advantages = []
            gae = 0
            
            for step in reversed(range(len(self.rewards))):
                if self.dones[step]:
                    delta = self.rewards[step] - values[step]
                    gae = delta
                else:
                    delta = self.rewards[step] + self.gamma * values[step + 1] - values[step]
                    gae = delta + self.gamma * gae_lambda * gae
                
                advantages.insert(0, gae)
            
            returns = [adv + val for adv, val in zip(advantages, self.values)]
        else:
            # N-step returns
            returns = []
            for i in range(len(self.rewards)):
                G = 0
                for j in range(i, min(i + self.n_steps, len(self.rewards))):
                    if self.dones[j]:
                        G += (self.gamma ** (j - i)) * self.rewards[j]
                        break
                    G += (self.gamma ** (j - i)) * self.rewards[j]
                else:
                    # Bootstrap with value function if episode didn't end
                    if i + self.n_steps < len(self.rewards):
                        G += (self.gamma ** self.n_steps) * values[i + self.n_steps]
                    else:
                        G += (self.gamma ** (len(self.rewards) - i)) * next_value
                
                returns.append(G)
            
            advantages = [ret - val for ret, val in zip(returns, self.values)]
        
        # Normalize advantages
        if normalize and len(advantages) > 1:
            advantages = np.array(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.tolist()
        
        return returns, advantages
    
    def get_batch(self):
        """Get all experiences as a batch"""
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'dones': self.dones
        }

class A2CAgent(BaseAgent):
    """
    Actor-Critic (A2C) Agent
    
    Implements synchronous advantage actor-critic with:
    - N-step returns for reduced variance
    - Generalized Advantage Estimation
    - Shared or separate actor-critic networks
    - Entropy regularization for exploration
    - Advanced gradient clipping and regularization
    """
    
    def __init__(self, observation_space, action_space, config: A2CConfig = None):
        if config is None:
            config = A2CConfig()
        
        super().__init__(observation_space, action_space, config)
        
        # A2C-specific attributes
        self.config: A2CConfig = config
        self.epsilon = config.epsilon_start
        self.experience_buffer = ExperienceBuffer(config.n_steps, config.gamma)
        
        # Training statistics
        self.policy_losses = deque(maxlen=1000)
        self.value_losses = deque(maxlen=1000)
        self.entropy_losses = deque(maxlen=1000)
        self.advantages = deque(maxlen=1000)
        
    def _initialize_algorithm(self):
        """Initialize A2C-specific components"""
        # Create actor-critic network
        self.actor_critic = A2CNetwork(
            self.observation_space, self.action_space, self.config
        ).to(self.device)
        
        # Create optimizer
        if self.config.use_rms_prop:
            self.optimizer = optim.RMSprop(
                self.actor_critic.parameters(),
                lr=self.config.learning_rate,
                alpha=self.config.alpha,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.actor_critic.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Learning rate scheduler
        if self.config.lr_schedule:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_schedule_step,
                gamma=self.config.lr_schedule_gamma
            )
        
        self.logger.info(f"Initialized A2C with {sum(p.numel() for p in self.actor_critic.parameters())} parameters")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_dist = self.actor_critic.get_action_distribution(state_tensor)
            value = self.actor_critic.get_value(state_tensor)
            
            if training:
                if self.config.exploration_method == 'entropy':
                    # Sample from policy distribution
                    action = action_dist.sample()
                elif self.config.exploration_method == 'epsilon':
                    # Epsilon-greedy exploration
                    if np.random.random() < self.epsilon:
                        action = torch.randint(0, self.action_space.n, (1,))
                    else:
                        action = action_dist.probs.argmax(dim=-1)
                else:
                    # Pure exploitation for noise-based exploration
                    action = action_dist.probs.argmax(dim=-1)
                
                log_prob = action_dist.log_prob(action)
                
                # Store for experience buffer
                self._last_action_info = {
                    'log_prob': log_prob.item(),
                    'value': value.item()
                }
            else:
                # Deterministic action for evaluation
                action = action_dist.probs.argmax(dim=-1)
        
        return action.item()
    
    def _process_experience(self, experience: Tuple) -> Optional[Dict[str, float]]:
        """Process experience and update when buffer is full"""
        state, action, reward, next_state, done = experience
        
        # Add to experience buffer
        if hasattr(self, '_last_action_info'):
            self.experience_buffer.add(
                state, action, reward,
                self._last_action_info['value'],
                self._last_action_info['log_prob'],
                done
            )
        
        # Update exploration parameter
        self._update_exploration()
        
        # Update when buffer is full or episode ends
        if len(self.experience_buffer.states) >= self.config.n_steps or done:
            # Get next value for bootstrapping
            with torch.no_grad():
                if done:
                    next_value = 0.0
                else:
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                    next_value = self.actor_critic.get_value(next_state_tensor).item()
            
            # Update networks
            update_info = self._update_networks(next_value)
            
            # Reset buffer
            self.experience_buffer.reset()
            
            return update_info
        
        return None
    
    def _update_exploration(self):
        """Update exploration parameters"""
        if self.config.exploration_method == 'epsilon':
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )
    
    def _update_networks(self, next_value: float) -> Dict[str, float]:
        """Update actor-critic networks"""
        if len(self.experience_buffer.states) == 0:
            return {}
        
        # Get batch data
        batch = self.experience_buffer.get_batch()
        
        # Compute returns and advantages
        returns, advantages = self.experience_buffer.compute_returns_and_advantages(
            next_value, self.config.use_gae, self.config.gae_lambda, 
            self.config.normalize_advantages
        )
        
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Forward pass
        log_probs, values, entropy = self.actor_critic.evaluate_actions(states, actions)
        
        # Compute losses
        # Policy loss (advantage-weighted policy gradient)
        policy_loss = -(log_probs * advantages_tensor.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns_tensor)
        
        # Entropy loss (for exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.config.value_loss_coef * value_loss + 
                     self.config.entropy_coef * entropy_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.config.max_grad_norm
            )
        
        self.optimizer.step()
        
        # Update learning rate
        if self.config.lr_schedule:
            self.scheduler.step()
        
        # Record statistics
        policy_loss_item = policy_loss.item()
        value_loss_item = value_loss.item()
        entropy_loss_item = entropy_loss.item()
        avg_advantage = advantages_tensor.mean().item()
        
        self.policy_losses.append(policy_loss_item)
        self.value_losses.append(value_loss_item)
        self.entropy_losses.append(entropy_loss_item)
        self.advantages.append(avg_advantage)
        
        return {
            'policy_loss': policy_loss_item,
            'value_loss': value_loss_item,
            'entropy_loss': entropy_loss_item,
            'total_loss': total_loss.item(),
            'average_advantage': avg_advantage,
            'entropy': entropy.mean().item()
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
            'epsilon': self.epsilon,
            'current_episode': self.current_episode,
            'best_score': self.best_score,
            'total_steps': self.total_steps
        }
        
        if self.config.lr_schedule:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.config.lr_schedule:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
        self.current_episode = checkpoint.get('current_episode', 0)
        self.best_score = checkpoint.get('best_score', float('-inf'))
        self.total_steps = checkpoint.get('total_steps', 0)
        
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
            'epsilon': self.epsilon,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        if self.policy_losses:
            stats['avg_policy_loss'] = np.mean(list(self.policy_losses)[-10:])
            
        if self.value_losses:
            stats['avg_value_loss'] = np.mean(list(self.value_losses)[-10:])
            
        if self.entropy_losses:
            stats['avg_entropy_loss'] = np.mean(list(self.entropy_losses)[-10:])
            
        if self.advantages:
            stats['avg_advantage'] = np.mean(list(self.advantages)[-10:])
        
        return stats