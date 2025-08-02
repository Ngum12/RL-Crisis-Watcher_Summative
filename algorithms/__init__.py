# Advanced Reinforcement Learning Algorithms Package
"""
Sophisticated implementations of state-of-the-art RL algorithms for Crisis Response AI:

- Deep Q-Network (DQN): Value-based learning with experience replay and target networks
- REINFORCE: Policy gradient method with baseline for variance reduction
- Proximal Policy Optimization (PPO): Advanced policy gradient with clipped objectives
- Actor-Critic (A2C): Combined value and policy learning with advantage estimation

Each algorithm includes:
- Professional network architectures optimized for the crisis domain
- Advanced hyperparameter tuning and optimization
- Comprehensive logging and monitoring
- Sophisticated exploration strategies
- Performance analysis and visualization
- Robust training stability mechanisms
"""

from .dqn import DQNAgent, DQNConfig
from .reinforce import REINFORCEAgent, REINFORCEConfig
from .ppo import PPOAgent, PPOConfig
from .actor_critic import A2CAgent, A2CConfig
from .base_agent import BaseAgent, AgentConfig
from .hyperparameter_tuner import HyperparameterTuner, TuningConfig

__all__ = [
    'DQNAgent',
    'DQNConfig',
    'REINFORCEAgent', 
    'REINFORCEConfig',
    'PPOAgent',
    'PPOConfig',
    'A2CAgent',
    'A2CConfig',
    'BaseAgent',
    'AgentConfig',
    'HyperparameterTuner',
    'TuningConfig'
]