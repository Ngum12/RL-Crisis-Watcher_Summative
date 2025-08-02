"""
Performance Evaluator for Crisis Response AI

Comprehensive evaluation system with multiple metrics:
- Episode performance analysis
- Statistical significance testing
- Convergence analysis
- Robustness evaluation
- Comparative analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import time
from scipy import stats
from collections import defaultdict

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    mean_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    median_reward: float
    success_rate: float
    convergence_episodes: int
    stability_score: float
    efficiency_score: float
    final_performance: float

class PerformanceEvaluator:
    """Comprehensive performance evaluation system"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.evaluation_history = defaultdict(list)
    
    def evaluate_agent(self, agent, environment, num_episodes: int = 100, 
                      render: bool = False) -> EvaluationMetrics:
        """Comprehensive agent evaluation"""
        episode_rewards = []
        episode_lengths = []
        success_episodes = 0
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0.0
            episode_length = 0
            
            max_steps = getattr(environment, 'time_horizon', 1000)
            
            for step in range(max_steps):
                action = agent.select_action(state, training=False)
                state, reward, done, info = environment.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if render and hasattr(environment, 'render'):
                    environment.render()
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Define success criteria (customize as needed)
            if episode_reward > 100:  # Threshold for success
                success_episodes += 1
        
        # Calculate metrics
        rewards_array = np.array(episode_rewards)
        
        metrics = EvaluationMetrics(
            mean_reward=float(np.mean(rewards_array)),
            std_reward=float(np.std(rewards_array)),
            max_reward=float(np.max(rewards_array)),
            min_reward=float(np.min(rewards_array)),
            median_reward=float(np.median(rewards_array)),
            success_rate=success_episodes / num_episodes,
            convergence_episodes=self._estimate_convergence(agent),
            stability_score=self._calculate_stability(rewards_array),
            efficiency_score=self._calculate_efficiency(episode_lengths),
            final_performance=float(np.mean(rewards_array[-10:]) if len(rewards_array) >= 10 else np.mean(rewards_array))
        )
        
        return metrics
    
    def _estimate_convergence(self, agent) -> int:
        """Estimate convergence episodes from training history"""
        if hasattr(agent, 'episode_rewards') and len(agent.episode_rewards) > 50:
            rewards = list(agent.episode_rewards)
            
            # Simple convergence detection: when performance stabilizes
            window_size = 50
            threshold = 0.05  # 5% improvement threshold
            
            for i in range(window_size, len(rewards)):
                recent_mean = np.mean(rewards[i-window_size:i])
                previous_mean = np.mean(rewards[i-2*window_size:i-window_size])
                
                if abs(recent_mean - previous_mean) / abs(previous_mean + 1e-8) < threshold:
                    return i
            
            return len(rewards)
        return -1
    
    def _calculate_stability(self, rewards: np.ndarray) -> float:
        """Calculate stability score (lower variance = higher stability)"""
        if len(rewards) < 2:
            return 0.0
        
        # Normalize by mean to get coefficient of variation
        cv = np.std(rewards) / (abs(np.mean(rewards)) + 1e-8)
        
        # Convert to stability score (0-1, where 1 is most stable)
        stability = 1.0 / (1.0 + cv)
        return float(stability)
    
    def _calculate_efficiency(self, episode_lengths: List[int]) -> float:
        """Calculate efficiency score based on episode lengths"""
        if not episode_lengths:
            return 0.0
        
        # Shorter episodes (faster solutions) are more efficient
        avg_length = np.mean(episode_lengths)
        max_length = max(episode_lengths)
        
        # Normalize to 0-1 scale
        efficiency = 1.0 - (avg_length / max_length)
        return float(efficiency)
    
    def compare_agents(self, agent_results: Dict[str, EvaluationMetrics]) -> Dict[str, Any]:
        """Compare multiple agents statistically"""
        comparison = {
            'rankings': {},
            'statistical_tests': {},
            'best_performer': None,
            'summary': {}
        }
        
        # Extract metrics for comparison
        metrics_data = defaultdict(dict)
        for agent_name, metrics in agent_results.items():
            metrics_data['mean_reward'][agent_name] = metrics.mean_reward
            metrics_data['success_rate'][agent_name] = metrics.success_rate
            metrics_data['stability_score'][agent_name] = metrics.stability_score
            metrics_data['efficiency_score'][agent_name] = metrics.efficiency_score
        
        # Rank agents by different metrics
        for metric_name, agent_scores in metrics_data.items():
            sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
            comparison['rankings'][metric_name] = [agent for agent, _ in sorted_agents]
        
        # Find overall best performer (weighted average of normalized metrics)
        weights = {'mean_reward': 0.4, 'success_rate': 0.3, 'stability_score': 0.2, 'efficiency_score': 0.1}
        overall_scores = defaultdict(float)
        
        for metric_name, agent_scores in metrics_data.items():
            if metric_name in weights:
                # Normalize scores to 0-1 range
                values = list(agent_scores.values())
                min_val, max_val = min(values), max(values)
                range_val = max_val - min_val if max_val != min_val else 1
                
                for agent_name, score in agent_scores.items():
                    normalized_score = (score - min_val) / range_val
                    overall_scores[agent_name] += weights[metric_name] * normalized_score
        
        best_agent = max(overall_scores.items(), key=lambda x: x[1])
        comparison['best_performer'] = best_agent[0]
        comparison['overall_scores'] = dict(overall_scores)
        
        return comparison