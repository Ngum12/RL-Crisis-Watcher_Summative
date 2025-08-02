"""
Hyperparameter Optimizer for Crisis Response AI

Advanced hyperparameter optimization using Optuna:
- Bayesian optimization for efficient search
- Algorithm-specific parameter spaces
- Multi-objective optimization
- Pruning for faster convergence
- Automated hyperparameter tuning
"""

import optuna
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import time

@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization"""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    optimization_time: float
    study_summary: Dict[str, Any]

class HyperparameterOptimizer:
    """Hyperparameter optimizer using Optuna"""
    
    def __init__(self, algorithm: str, environment, visualizer):
        self.algorithm = algorithm
        self.environment = environment
        self.visualizer = visualizer
        
        # Setup logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
    def optimize(self, n_trials: int = 100, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Optimize hyperparameters for the algorithm"""
        
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.algorithm}_optimization',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)
        )
        
        start_time = time.time()
        
        try:
            study.optimize(
                self._objective_function,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print("Optimization interrupted by user")
        
        optimization_time = time.time() - start_time
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_time': optimization_time
        }
    
    def _objective_function(self, trial) -> float:
        """Objective function for optimization"""
        
        # Get algorithm-specific parameter suggestions
        if self.algorithm == 'DQN':
            params = self._suggest_dqn_params(trial)
        elif self.algorithm == 'REINFORCE':
            params = self._suggest_reinforce_params(trial)
        elif self.algorithm == 'PPO':
            params = self._suggest_ppo_params(trial)
        elif self.algorithm == 'A2C':
            params = self._suggest_a2c_params(trial)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Create and train agent with suggested parameters
        from algorithms.dqn import DQNAgent, DQNConfig
        from algorithms.reinforce import REINFORCEAgent, REINFORCEConfig
        from algorithms.ppo import PPOAgent, PPOConfig
        from algorithms.actor_critic import A2CAgent, A2CConfig
        
        config_classes = {
            'DQN': DQNConfig,
            'REINFORCE': REINFORCEConfig,
            'PPO': PPOConfig,
            'A2C': A2CConfig
        }
        
        agent_classes = {
            'DQN': DQNAgent,
            'REINFORCE': REINFORCEAgent,
            'PPO': PPOAgent,
            'A2C': A2CAgent
        }
        
        # Create config with suggested parameters
        config_class = config_classes[self.algorithm]
        config = config_class(**params)
        
        # Reduce training time for optimization
        config.max_episodes = 100  # Quick evaluation
        config.log_frequency = 50
        config.save_frequency = 999999  # Disable saving during optimization
        
        # Create agent
        agent_class = agent_classes[self.algorithm]
        agent = agent_class(
            self.environment.observation_space,
            self.environment.action_space,
            config
        )
        
        try:
            # Train agent
            training_stats = agent.train(self.environment, num_episodes=100)
            
            # Calculate objective score
            if 'episode_rewards' in training_stats:
                rewards = training_stats['episode_rewards']
                
                # Use mean of last 20 episodes as score
                score = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
                
                # Add stability bonus (penalize high variance)
                if len(rewards) >= 10:
                    stability = 1.0 / (1.0 + np.std(rewards[-20:]))
                    score += stability * 10  # Small stability bonus
                
            else:
                score = agent.best_score if hasattr(agent, 'best_score') else 0.0
            
            # Pruning check
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return float(score)
            
        except Exception as e:
            print(f"Trial failed: {str(e)}")
            return -float('inf')
    
    def _suggest_dqn_params(self, trial) -> Dict[str, Any]:
        """Suggest DQN hyperparameters"""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'gamma': trial.suggest_uniform('gamma', 0.9, 0.999),
            'epsilon_decay': trial.suggest_uniform('epsilon_decay', 0.99, 0.999),
            'target_update_frequency': trial.suggest_categorical('target_update_frequency', [100, 500, 1000, 2000]),
            'hidden_layers': trial.suggest_categorical('hidden_layers', [
                [256, 128], [512, 256], [512, 256, 128], [256, 256, 128]
            ]),
            'memory_size': trial.suggest_categorical('memory_size', [10000, 50000, 100000]),
            'double_dqn': trial.suggest_categorical('double_dqn', [True, False]),
            'dueling_dqn': trial.suggest_categorical('dueling_dqn', [True, False])
        }
    
    def _suggest_reinforce_params(self, trial) -> Dict[str, Any]:
        """Suggest REINFORCE hyperparameters"""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'gamma': trial.suggest_uniform('gamma', 0.9, 0.999),
            'n_trajectories': trial.suggest_categorical('n_trajectories', [5, 10, 20]),
            'entropy_coef': trial.suggest_loguniform('entropy_coef', 1e-4, 1e-1),
            'hidden_layers': trial.suggest_categorical('hidden_layers', [
                [128, 64], [256, 128], [256, 128, 64], [128, 128]
            ]),
            'use_baseline': trial.suggest_categorical('use_baseline', [True, False]),
            'normalize_advantages': trial.suggest_categorical('normalize_advantages', [True, False]),
            'advantage_method': trial.suggest_categorical('advantage_method', ['returns', 'gae'])
        }
    
    def _suggest_ppo_params(self, trial) -> Dict[str, Any]:
        """Suggest PPO hyperparameters"""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048]),
            'n_epochs': trial.suggest_categorical('n_epochs', [3, 5, 10, 20]),
            'minibatch_size': trial.suggest_categorical('minibatch_size', [32, 64, 128]),
            'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.3),
            'gamma': trial.suggest_uniform('gamma', 0.9, 0.999),
            'gae_lambda': trial.suggest_uniform('gae_lambda', 0.9, 0.99),
            'entropy_coef': trial.suggest_loguniform('entropy_coef', 1e-4, 1e-1),
            'value_loss_coef': trial.suggest_uniform('value_loss_coef', 0.25, 1.0),
            'hidden_layers': trial.suggest_categorical('hidden_layers', [
                [128, 128], [256, 256], [512, 256], [256, 128]
            ])
        }
    
    def _suggest_a2c_params(self, trial) -> Dict[str, Any]:
        """Suggest A2C hyperparameters"""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'n_steps': trial.suggest_categorical('n_steps', [5, 10, 20]),
            'gamma': trial.suggest_uniform('gamma', 0.9, 0.999),
            'gae_lambda': trial.suggest_uniform('gae_lambda', 0.9, 0.99),
            'entropy_coef': trial.suggest_loguniform('entropy_coef', 1e-4, 1e-1),
            'value_loss_coef': trial.suggest_uniform('value_loss_coef', 0.25, 1.0),
            'hidden_layers': trial.suggest_categorical('hidden_layers', [
                [128, 64], [256, 128], [256, 256], [128, 128]
            ]),
            'shared_network': trial.suggest_categorical('shared_network', [True, False]),
            'use_rms_prop': trial.suggest_categorical('use_rms_prop', [True, False])
        }