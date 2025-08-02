"""
Performance Charts and Training Visualizations

Advanced charting capabilities for the Crisis Response AI including:
- Training progress visualization
- Performance metrics charts
- Algorithm comparison plots
- Real-time training monitoring
- Statistical analysis displays
"""

import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd

# Set style for professional plots
plt.style.use('dark_background')
sns.set_palette("husl")

class PerformanceCharts:
    """Generate performance charts for RL training analysis"""
    
    def __init__(self, style: str = 'dark'):
        self.style = style
        self.colors = {
            'DQN': '#FF6B6B',
            'PPO': '#4ECDC4', 
            'REINFORCE': '#45B7D1',
            'A2C': '#FFA07A'
        }
        
    def plot_training_progress(self, training_data: Dict[str, List[float]], 
                             save_path: str = None, show_plot: bool = True) -> plt.Figure:
        """Plot training progress for multiple algorithms"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Crisis Response AI - Training Progress', fontsize=16, fontweight='bold')
        
        # Episode rewards
        ax1.set_title('Episode Rewards Over Time')
        for algo, rewards in training_data.items():
            if len(rewards) > 0:
                episodes = range(len(rewards))
                ax1.plot(episodes, rewards, label=algo, color=self.colors.get(algo, '#FFFFFF'), linewidth=2)
                
                # Add moving average
                if len(rewards) > 20:
                    moving_avg = pd.Series(rewards).rolling(window=20).mean()
                    ax1.plot(episodes, moving_avg, '--', alpha=0.7, color=self.colors.get(algo, '#FFFFFF'))
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Cumulative Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Average reward comparison
        ax2.set_title('Average Reward Comparison')
        algos = list(training_data.keys())
        avg_rewards = [np.mean(training_data[algo]) if len(training_data[algo]) > 0 else 0 for algo in algos]
        bars = ax2.bar(algos, avg_rewards, color=[self.colors.get(algo, '#FFFFFF') for algo in algos])
        ax2.set_ylabel('Average Reward')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_rewards):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_rewards)*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Learning curves (smoothed)
        ax3.set_title('Learning Curves (Smoothed)')
        for algo, rewards in training_data.items():
            if len(rewards) > 10:
                smoothed = pd.Series(rewards).rolling(window=10, min_periods=1).mean()
                ax3.plot(range(len(smoothed)), smoothed, label=algo, 
                        color=self.colors.get(algo, '#FFFFFF'), linewidth=2)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Smoothed Reward')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Success rate over time (if available)
        ax4.set_title('Performance Stability')
        for algo, rewards in training_data.items():
            if len(rewards) > 50:
                # Calculate rolling standard deviation as stability metric
                rolling_std = pd.Series(rewards).rolling(window=50).std()
                ax4.plot(range(len(rolling_std)), rolling_std, label=f'{algo} (Std Dev)', 
                        color=self.colors.get(algo, '#FFFFFF'), linewidth=2)
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward Standard Deviation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_hyperparameter_analysis(self, hyperparameter_results: Dict[str, Dict[str, List[float]]], 
                                   save_path: str = None) -> plt.Figure:
        """Plot hyperparameter sensitivity analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        # Common hyperparameters to analyze
        hyperparams = ['learning_rate', 'gamma', 'batch_size', 'epsilon']
        
        for i, hyperparam in enumerate(hyperparams):
            if i < len(axes):
                ax = axes[i]
                ax.set_title(f'{hyperparam.replace("_", " ").title()} Sensitivity')
                
                for algo, results in hyperparameter_results.items():
                    if hyperparam in results:
                        values = results[hyperparam]
                        rewards = results.get('rewards', [])
                        if len(values) == len(rewards):
                            ax.scatter(values, rewards, label=algo, 
                                     color=self.colors.get(algo, '#FFFFFF'), alpha=0.7, s=50)
                
                ax.set_xlabel(hyperparam.replace('_', ' ').title())
                ax.set_ylabel('Average Reward')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        
        return fig
    
    def plot_convergence_analysis(self, training_data: Dict[str, List[float]], 
                                save_path: str = None) -> plt.Figure:
        """Analyze convergence characteristics of different algorithms"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
        
        # Episodes to convergence
        convergence_episodes = {}
        convergence_threshold = 0.95  # 95% of final performance
        
        for algo, rewards in training_data.items():
            if len(rewards) > 100:
                final_performance = np.mean(rewards[-50:])  # Last 50 episodes
                threshold_value = final_performance * convergence_threshold
                
                # Find first episode where performance consistently exceeds threshold
                for i in range(50, len(rewards)):
                    if np.mean(rewards[i-10:i+10]) >= threshold_value:
                        convergence_episodes[algo] = i
                        break
                else:
                    convergence_episodes[algo] = len(rewards)
        
        # Plot convergence episodes
        algos = list(convergence_episodes.keys())
        episodes = list(convergence_episodes.values())
        bars = ax1.bar(algos, episodes, color=[self.colors.get(algo, '#FFFFFF') for algo in algos])
        ax1.set_title('Episodes to Convergence')
        ax1.set_ylabel('Episodes')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, episodes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(episodes)*0.01,
                    f'{value}', ha='center', va='bottom')
        
        # Final performance comparison
        final_rewards = {}
        for algo, rewards in training_data.items():
            if len(rewards) > 50:
                final_rewards[algo] = np.mean(rewards[-50:])
        
        algos = list(final_rewards.keys())
        final_values = list(final_rewards.values())
        bars = ax2.bar(algos, final_values, color=[self.colors.get(algo, '#FFFFFF') for algo in algos])
        ax2.set_title('Final Performance (Last 50 Episodes)')
        ax2.set_ylabel('Average Reward')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, final_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        
        return fig

class TrainingVisualizer:
    """Real-time training visualization and monitoring"""
    
    def __init__(self):
        self.episode_data = defaultdict(list)
        self.step_data = defaultdict(list)
        self.hyperparameter_data = defaultdict(dict)
        
    def log_episode(self, algorithm: str, episode: int, reward: float, 
                   length: int, additional_metrics: Dict[str, float] = None):
        """Log episode data for visualization"""
        self.episode_data[f'{algorithm}_reward'].append(reward)
        self.episode_data[f'{algorithm}_length'].append(length)
        self.episode_data[f'{algorithm}_episode'].append(episode)
        
        if additional_metrics:
            for metric, value in additional_metrics.items():
                self.episode_data[f'{algorithm}_{metric}'].append(value)
    
    def log_step(self, algorithm: str, step: int, loss: float = None, 
                q_value: float = None, additional_metrics: Dict[str, float] = None):
        """Log step-level data for detailed analysis"""
        self.step_data[f'{algorithm}_step'].append(step)
        
        if loss is not None:
            self.step_data[f'{algorithm}_loss'].append(loss)
        if q_value is not None:
            self.step_data[f'{algorithm}_q_value'].append(q_value)
        
        if additional_metrics:
            for metric, value in additional_metrics.items():
                self.step_data[f'{algorithm}_{metric}'].append(value)
    
    def log_hyperparameters(self, algorithm: str, hyperparams: Dict[str, Any]):
        """Log hyperparameter configuration"""
        self.hyperparameter_data[algorithm] = hyperparams
    
    def generate_training_report(self, save_dir: str = 'reports/'):
        """Generate comprehensive training report with all visualizations"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract algorithm names
        algorithms = set()
        for key in self.episode_data.keys():
            algo = key.split('_')[0]
            algorithms.add(algo)
        
        # Prepare data for charting
        training_data = {}
        for algo in algorithms:
            if f'{algo}_reward' in self.episode_data:
                training_data[algo] = self.episode_data[f'{algo}_reward']
        
        # Generate charts
        charts = PerformanceCharts()
        
        # Training progress
        if training_data:
            fig1 = charts.plot_training_progress(training_data, 
                                               save_path=f'{save_dir}/training_progress.png',
                                               show_plot=False)
            plt.close(fig1)
        
        # Convergence analysis
        if training_data:
            fig2 = charts.plot_convergence_analysis(training_data,
                                                   save_path=f'{save_dir}/convergence_analysis.png')
            plt.close(fig2)
        
        # Generate detailed statistics
        self._generate_statistics_report(algorithms, save_dir)
        
        return f"Training report generated in {save_dir}"
    
    def _generate_statistics_report(self, algorithms: List[str], save_dir: str):
        """Generate detailed statistics report"""
        stats = {}
        
        for algo in algorithms:
            if f'{algo}_reward' in self.episode_data:
                rewards = self.episode_data[f'{algo}_reward']
                lengths = self.episode_data.get(f'{algo}_length', [])
                
                stats[algo] = {
                    'total_episodes': len(rewards),
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'max_reward': np.max(rewards),
                    'min_reward': np.min(rewards),
                    'final_100_mean': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
                    'mean_episode_length': np.mean(lengths) if lengths else 0,
                    'hyperparameters': self.hyperparameter_data.get(algo, {})
                }
        
        # Save statistics as JSON
        import json
        with open(f'{save_dir}/training_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Generate markdown report
        with open(f'{save_dir}/training_report.md', 'w') as f:
            f.write("# Crisis Response AI - Training Report\n\n")
            f.write("## Algorithm Performance Summary\n\n")
            
            for algo, data in stats.items():
                f.write(f"### {algo}\n")
                f.write(f"- **Total Episodes**: {data['total_episodes']}\n")
                f.write(f"- **Mean Reward**: {data['mean_reward']:.2f} ± {data['std_reward']:.2f}\n")
                f.write(f"- **Best Reward**: {data['max_reward']:.2f}\n")
                f.write(f"- **Final 100 Episodes Mean**: {data['final_100_mean']:.2f}\n")
                f.write(f"- **Average Episode Length**: {data['mean_episode_length']:.1f}\n")
                
                if data['hyperparameters']:
                    f.write(f"- **Hyperparameters**:\n")
                    for param, value in data['hyperparameters'].items():
                        f.write(f"  - {param}: {value}\n")
                f.write("\n")
    
    def get_current_performance(self, algorithm: str) -> Dict[str, float]:
        """Get current performance metrics for an algorithm"""
        if f'{algorithm}_reward' not in self.episode_data:
            return {}
        
        rewards = self.episode_data[f'{algorithm}_reward']
        if len(rewards) == 0:
            return {}
        
        return {
            'latest_reward': rewards[-1],
            'mean_reward': np.mean(rewards),
            'mean_last_10': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
            'mean_last_100': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
            'total_episodes': len(rewards)
        }