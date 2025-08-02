"""
Deep Q-Network (DQN) Implementation

Advanced DQN with state-of-the-art features:
- Experience Replay with prioritized sampling
- Target Networks for stable training
- Double DQN to reduce overestimation bias
- Dueling Network Architecture for better value estimation
- Noisy Networks for better exploration
- Rainbow DQN improvements
- Sophisticated exploration strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import random
from collections import deque
import math

from .base_agent import BaseAgent, AgentConfig, Memory, create_network

@dataclass
class DQNConfig(AgentConfig):
    """Configuration for DQN Agent"""
    # DQN-specific parameters
    memory_size: int = 100000
    min_memory_size: int = 1000
    target_update_frequency: int = 1000
    double_dqn: bool = True
    dueling_dqn: bool = True
    prioritized_replay: bool = True
    noisy_networks: bool = True
    
    # Learning parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    gamma: float = 0.99
    tau: float = 0.005  # Soft update parameter
    
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    epsilon_decay_method: str = 'exponential'  # 'exponential', 'linear', 'schedule'
    
    # Prioritized replay parameters
    alpha: float = 0.6  # Prioritization exponent
    beta_start: float = 0.4  # Importance sampling exponent
    beta_frames: int = 100000  # Frames to anneal beta to 1.0
    
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = 'relu'
    dropout_rate: float = 0.1
    
    # Advanced features
    gradient_clipping: float = 10.0
    weight_decay: float = 1e-5
    lr_schedule: bool = True
    lr_schedule_step: int = 10000
    lr_schedule_gamma: float = 0.9

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    
    def __init__(self, input_size: int, output_size: int, sigma_init: float = 0.017):
        super(NoisyLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(output_size, input_size))
        self.weight_sigma = nn.Parameter(torch.empty(output_size, input_size))
        self.bias_mu = nn.Parameter(torch.empty(output_size))
        self.bias_sigma = nn.Parameter(torch.empty(output_size))
        
        # Noise buffers
        self.register_buffer('weight_epsilon', torch.empty(output_size, input_size))
        self.register_buffer('bias_epsilon', torch.empty(output_size))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.input_size)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.input_size))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.output_size))
    
    def reset_noise(self):
        """Reset noise buffers"""
        epsilon_in = self._scale_noise(self.input_size)
        epsilon_out = self._scale_noise(self.output_size)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy parameters"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)

class DuelingDQN(nn.Module):
    """Dueling DQN architecture"""
    
    def __init__(self, input_size: int, output_size: int, config: DQNConfig):
        super(DuelingDQN, self).__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        
        # Shared feature extractor
        self.feature_layer = create_network(
            input_size, 
            config.hidden_layers[-1],
            config.hidden_layers[:-1],
            config.activation,
            config.dropout_rate
        )
        
        # Value stream
        if config.noisy_networks:
            self.value_stream = nn.Sequential(
                NoisyLinear(config.hidden_layers[-1], config.hidden_layers[-1]),
                nn.ReLU(),
                NoisyLinear(config.hidden_layers[-1], 1)
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(config.hidden_layers[-1], config.hidden_layers[-1]),
                nn.ReLU(),
                nn.Linear(config.hidden_layers[-1], 1)
            )
        
        # Advantage stream
        if config.noisy_networks:
            self.advantage_stream = nn.Sequential(
                NoisyLinear(config.hidden_layers[-1], config.hidden_layers[-1]),
                nn.ReLU(),
                NoisyLinear(config.hidden_layers[-1], output_size)
            )
        else:
            self.advantage_stream = nn.Sequential(
                nn.Linear(config.hidden_layers[-1], config.hidden_layers[-1]),
                nn.ReLU(),
                nn.Linear(config.hidden_layers[-1], output_size)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        if self.config.noisy_networks:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

class RegularDQN(nn.Module):
    """Regular DQN architecture"""
    
    def __init__(self, input_size: int, output_size: int, config: DQNConfig):
        super(RegularDQN, self).__init__()
        self.config = config
        
        if config.noisy_networks:
            # Build network with noisy layers
            layers = []
            current_size = input_size
            
            for hidden_size in config.hidden_layers:
                layers.append(NoisyLinear(current_size, hidden_size))
                if config.activation == 'relu':
                    layers.append(nn.ReLU())
                elif config.activation == 'tanh':
                    layers.append(nn.Tanh())
                
                if config.dropout_rate > 0:
                    layers.append(nn.Dropout(config.dropout_rate))
                
                current_size = hidden_size
            
            layers.append(NoisyLinear(current_size, output_size))
            self.network = nn.Sequential(*layers)
        else:
            self.network = create_network(
                input_size, output_size, config.hidden_layers,
                config.activation, config.dropout_rate
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        if self.config.noisy_networks:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

class PrioritizedReplayMemory:
    """Prioritized Experience Replay"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, experience: Tuple):
        """Add experience with maximum priority"""
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
            self.priorities.append(self.max_priority)
        else:
            self.memory[self.position] = experience
            self.priorities[self.position] = self.max_priority
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Tuple], List[int], torch.Tensor]:
        """Sample batch with importance sampling weights"""
        if len(self.memory) < batch_size:
            return [], [], torch.tensor([])
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        experiences = [self.memory[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = torch.FloatTensor(weights)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for given indices"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent(BaseAgent):
    """
    Advanced Deep Q-Network Agent
    
    Implements state-of-the-art DQN with multiple improvements:
    - Double DQN for reduced overestimation
    - Dueling architecture for better value estimation
    - Prioritized replay for efficient learning
    - Noisy networks for exploration
    - Sophisticated exploration strategies
    """
    
    def __init__(self, observation_space, action_space, config: DQNConfig = None):
        if config is None:
            config = DQNConfig()
        
        super().__init__(observation_space, action_space, config)
        
        # DQN-specific attributes
        self.config: DQNConfig = config
        self.epsilon = config.epsilon_start
        self.beta = config.beta_start
        self.steps_done = 0
        
    def _initialize_algorithm(self):
        """Initialize DQN-specific components"""
        input_size = np.prod(self.observation_space.shape)
        output_size = self.action_space.n
        
        # Create networks
        if self.config.dueling_dqn:
            self.q_network = DuelingDQN(input_size, output_size, self.config).to(self.device)
            self.target_network = DuelingDQN(input_size, output_size, self.config).to(self.device)
        else:
            self.q_network = RegularDQN(input_size, output_size, self.config).to(self.device)
            self.target_network = RegularDQN(input_size, output_size, self.config).to(self.device)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        if self.config.lr_schedule:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_schedule_step,
                gamma=self.config.lr_schedule_gamma
            )
        
        # Experience replay
        if self.config.prioritized_replay:
            self.memory = PrioritizedReplayMemory(
                self.config.memory_size,
                self.config.alpha
            )
        else:
            self.memory = Memory(self.config.memory_size)
        
        self.logger.info(f"Initialized DQN with {sum(p.numel() for p in self.q_network.parameters())} parameters")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy or noisy networks"""
        if not training or (self.config.noisy_networks and training):
            # Use noisy networks for exploration during training
            # Or deterministic policy during evaluation
            return self._greedy_action(state)
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_space.n)
        else:
            return self._greedy_action(state)
    
    def _greedy_action(self, state: np.ndarray) -> int:
        """Select greedy action"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Reset noise for noisy networks
            if self.config.noisy_networks:
                self.q_network.reset_noise()
            
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
            
        return action
    
    def _process_experience(self, experience: Tuple) -> Optional[Dict[str, float]]:
        """Process experience and update network if needed"""
        self.memory.push(experience)
        self.steps_done += 1
        
        # Update exploration parameters
        self._update_exploration()
        
        # Update network
        if (len(self.memory) >= self.config.min_memory_size and 
            self.steps_done % self.config.update_frequency == 0):
            
            loss = self._update_network()
            
            # Update target network
            if self.steps_done % self.config.target_update_frequency == 0:
                self._update_target_network()
            
            return {'loss': loss}
        
        return None
    
    def _update_exploration(self):
        """Update exploration parameters"""
        if self.config.epsilon_decay_method == 'exponential':
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )
        elif self.config.epsilon_decay_method == 'linear':
            decay_rate = (self.config.epsilon_start - self.config.epsilon_end) / self.config.max_episodes
            self.epsilon = max(
                self.config.epsilon_end,
                self.config.epsilon_start - decay_rate * self.current_episode
            )
        
        # Update beta for prioritized replay
        if self.config.prioritized_replay:
            self.beta = min(1.0, self.config.beta_start + 
                          (1.0 - self.config.beta_start) * self.steps_done / self.config.beta_frames)
    
    def _update_network(self) -> float:
        """Update Q-network using sampled experiences"""
        if self.config.prioritized_replay:
            experiences, indices, weights = self.memory.sample(self.config.batch_size, self.beta)
            weights = weights.to(self.device)
        else:
            experiences = self.memory.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size).to(self.device)
            indices = None
        
        if not experiences:
            return 0.0
        
        # Unpack experiences
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                # Regular DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.config.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        
        if self.config.prioritized_replay:
            # Prioritized replay: weight the loss by importance sampling weights
            loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
            
            # Update priorities
            priorities = td_errors.abs().detach().cpu().numpy().flatten()
            priorities = np.clip(priorities, 1e-6, None)  # Avoid zero priorities
            self.memory.update_priorities(indices, priorities)
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clipping)
        
        self.optimizer.step()
        
        # Learning rate scheduling
        if self.config.lr_schedule:
            self.scheduler.step()
        
        return loss.item()
    
    def _update_target_network(self):
        """Update target network using soft update"""
        for target_param, main_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.config.tau * main_param.data + (1.0 - self.config.tau) * target_param.data
            )
    
    def update(self, experiences: List[Tuple]) -> Dict[str, float]:
        """Batch update for compatibility with base class"""
        for experience in experiences:
            self.memory.push(experience)
        
        if len(self.memory) >= self.config.min_memory_size:
            loss = self._update_network()
            return {'loss': loss}
        
        return {}
    
    def save_model(self, filepath: str):
        """Save model and training state"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'epsilon': self.epsilon,
            'beta': self.beta,
            'steps_done': self.steps_done,
            'current_episode': self.current_episode,
            'best_score': self.best_score
        }
        
        if self.config.lr_schedule:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.config.lr_schedule:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
        self.beta = checkpoint.get('beta', self.config.beta_start)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.current_episode = checkpoint.get('current_episode', 0)
        self.best_score = checkpoint.get('best_score', float('-inf'))
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state (for analysis)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def get_exploration_stats(self) -> Dict[str, float]:
        """Get current exploration statistics"""
        return {
            'epsilon': self.epsilon,
            'beta': self.beta if self.config.prioritized_replay else 0.0,
            'steps_done': self.steps_done,
            'memory_size': len(self.memory)
        }