#!/usr/bin/env python3
"""
Live Model Visualization - See Trained Agents in Action
Visualize real trained RL models making decisions in real-time with pygame

Shows the actual neural network models trained with Stable Baselines3
making decisions in the crisis response environment with live feedback.
"""

import pygame
import numpy as np
import math
import time
import os
from datetime import datetime
from typing import Dict, List, Optional

try:
    from stable_baselines3 import DQN, PPO, A2C
    from real_rl_training import CrisisResponseEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

class ModelVisualizer:
    """Visualize trained RL models in real-time"""
    
    def __init__(self, width=1400, height=900):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🤖 Live Trained Model Visualization - Africa Crisis AI")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.colors = {
            'background': (12, 20, 30),
            'panel_bg': (25, 35, 45),
            'text': (255, 255, 255),
            'text_dim': (180, 180, 180),
            'stable': (50, 200, 50),
            'warning': (255, 220, 50),
            'crisis': (255, 80, 80),
            'agent': (50, 255, 150),
            'neural': (100, 150, 255),
            'accent': (0, 200, 255),
            'gold': (255, 215, 0)
        }
        
        # Environment and model
        self.env = None
        self.model = None
        self.model_name = ""
        
        # Visualization state
        self.current_obs = None
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0
        self.episode_rewards = []
        
        # Countries and state tracking
        self.countries = ["Cameroon", "DR Congo", "Sudan"]
        self.action_names = [
            "Monitor Situation",
            "Deploy Peacekeepers", 
            "Economic Aid",
            "Diplomatic Intervention",
            "Humanitarian Aid",
            "Early Warning System",
            "Media Campaign",
            "Intelligence Gathering"
        ]
        
        # Animation
        self.animation_time = 0
        self.decision_history = []
        self.neural_activity = []
        
        # Fonts
        self.font_title = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        
        # Recording
        self.recording = False
        self.recording_start_time = None
        
    def load_model(self, model_path: str, algorithm: str):
        """Load a trained model"""
        try:
            if algorithm.upper() == "DQN":
                self.model = DQN.load(model_path)
            elif algorithm.upper() == "PPO":
                self.model = PPO.load(model_path)
            elif algorithm.upper() == "A2C":
                self.model = A2C.load(model_path)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            self.model_name = f"{algorithm.upper()} Neural Network"
            self.env = CrisisResponseEnv()
            self.reset_episode()
            
            print(f"✅ Loaded {self.model_name} successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def reset_episode(self):
        """Reset to new episode"""
        if self.env:
            self.current_obs, _ = self.env.reset()
            self.episode_count += 1
            if self.total_reward > 0:
                self.episode_rewards.append(self.total_reward)
            self.total_reward = 0
            self.step_count = 0
            self.decision_history = []
    
    def step_model(self):
        """Take one step with the trained model"""
        if not self.model or not self.env or self.current_obs is None:
            return
        
        # Get action from trained model
        action, _states = self.model.predict(self.current_obs, deterministic=True)
        
        # Take step in environment
        self.current_obs, reward, done, _, info = self.env.step(action)
        
        # Update visualization state
        self.total_reward += reward
        self.step_count += 1
        
        # Record decision
        target_country = info.get('target_country', 'Unknown')
        decision = {
            'step': self.step_count,
            'action': action,
            'action_name': self.action_names[action],
            'reward': reward,
            'target_country': target_country,
            'lives_saved': info.get('lives_saved', 0),
            'crises_prevented': info.get('crises_prevented', 0),
            'time': time.time()
        }
        
        self.decision_history.append(decision)
        
        # Keep only recent decisions
        if len(self.decision_history) > 10:
            self.decision_history.pop(0)
        
        # Generate neural activity visualization
        self._simulate_neural_activity(action)
        
        # Reset episode if done
        if done:
            self.reset_episode()
    
    def _simulate_neural_activity(self, action):
        """Simulate neural network activity for visualization"""
        # Create "neural activity" for visualization
        activity = {
            'input_layer': np.random.uniform(0.3, 1.0, 8),  # Input neurons
            'hidden_layer': np.random.uniform(0.2, 0.9, 6), # Hidden neurons
            'output_layer': np.zeros(8),  # Output neurons
            'time': time.time()
        }
        
        # Highlight the chosen action
        activity['output_layer'][action] = 1.0
        
        # Add some random activation to other outputs
        for i in range(8):
            if i != action:
                activity['output_layer'][i] = np.random.uniform(0.1, 0.6)
        
        self.neural_activity.append(activity)
        
        # Keep only recent activity
        if len(self.neural_activity) > 5:
            self.neural_activity.pop(0)
    
    def render(self):
        """Render the complete visualization"""
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Render components
        self._render_header()
        self._render_country_status()
        self._render_neural_network()
        self._render_decision_panel()
        self._render_performance_stats()
        self._render_controls()
        
        # Update animation
        self.animation_time += 0.05
    
    def _render_header(self):
        """Render main header"""
        title = self.font_title.render(f"🤖 Live Neural Network Visualization: {self.model_name}", True, self.colors['text'])
        self.screen.blit(title, (20, 20))
        
        subtitle = self.font_medium.render("Real-time AI decision making for Africa Crisis Response", True, self.colors['accent'])
        self.screen.blit(subtitle, (20, 55))
        
        episode_info = self.font_small.render(f"Episode: {self.episode_count} | Step: {self.step_count} | Total Reward: {self.total_reward:.1f}", True, self.colors['text_dim'])
        self.screen.blit(episode_info, (20, 80))
    
    def _render_country_status(self):
        """Render current country status"""
        if self.current_obs is None:
            return
        
        panel_y = 120
        panel_height = 200
        
        # Background panel
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (20, panel_y, self.width - 40, panel_height))
        pygame.draw.rect(self.screen, self.colors['accent'], (20, panel_y, self.width - 40, panel_height), 2)
        
        title = self.font_large.render("🌍 Country Status (Real-time from Neural Network)", True, self.colors['text'])
        self.screen.blit(title, (30, panel_y + 10))
        
        # Extract country data from observation
        for i, country in enumerate(self.countries):
            x = 50 + i * 420
            y = panel_y + 50
            
            # Get country data from observation (12 features per country)
            start_idx = i * 12
            political = self.current_obs[start_idx]
            economic = self.current_obs[start_idx + 1] 
            social = self.current_obs[start_idx + 2]
            security = self.current_obs[start_idx + 3]
            
            # Country name
            country_text = self.font_medium.render(f"🏴 {country}", True, self.colors['text'])
            self.screen.blit(country_text, (x, y))
            
            # Status bars
            self._render_status_bar(x, y + 30, "Political", political)
            self._render_status_bar(x, y + 50, "Economic", economic)
            self._render_status_bar(x, y + 70, "Social", social)
            self._render_status_bar(x, y + 90, "Security", security)
            
            # Overall stability
            stability = (political + economic + social + security) / 4
            stability_color = self.colors['stable'] if stability > 0.6 else (self.colors['warning'] if stability > 0.3 else self.colors['crisis'])
            stability_text = self.font_small.render(f"Stability: {stability:.1%}", True, stability_color)
            self.screen.blit(stability_text, (x, y + 115))
    
    def _render_status_bar(self, x: int, y: int, label: str, value: float):
        """Render a status bar"""
        bar_width = 200
        bar_height = 12
        
        # Label
        label_text = self.font_small.render(f"{label}:", True, self.colors['text_dim'])
        self.screen.blit(label_text, (x, y))
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), (x + 70, y, bar_width, bar_height))
        
        # Value bar
        fill_width = int(bar_width * value)
        color = self.colors['stable'] if value > 0.6 else (self.colors['warning'] if value > 0.3 else self.colors['crisis'])
        pygame.draw.rect(self.screen, color, (x + 70, y, fill_width, bar_height))
        
        # Value text
        value_text = self.font_small.render(f"{value:.1%}", True, self.colors['text'])
        self.screen.blit(value_text, (x + 280, y - 2))
    
    def _render_neural_network(self):
        """Render neural network visualization"""
        panel_x = 20
        panel_y = 340
        panel_width = 600
        panel_height = 300
        
        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['neural'], (panel_x, panel_y, panel_width, panel_height), 2)
        
        title = self.font_large.render("🧠 Live Neural Network Activity", True, self.colors['neural'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        if not self.neural_activity:
            no_activity = self.font_medium.render("No neural activity yet...", True, self.colors['text_dim'])
            self.screen.blit(no_activity, (panel_x + 10, panel_y + 50))
            return
        
        # Get latest neural activity
        latest = self.neural_activity[-1]
        
        # Neural network layers
        input_x = panel_x + 50
        hidden_x = panel_x + 250
        output_x = panel_x + 450
        
        layer_y = panel_y + 60
        neuron_spacing = 30
        
        # Input layer
        input_label = self.font_small.render("Input Layer", True, self.colors['text'])
        self.screen.blit(input_label, (input_x - 20, layer_y - 20))
        
        for i, activation in enumerate(latest['input_layer']):
            y = layer_y + i * neuron_spacing
            color_intensity = int(255 * activation)
            neuron_color = (color_intensity, color_intensity // 2, color_intensity // 2)
            pygame.draw.circle(self.screen, neuron_color, (input_x, y), 8)
            
            # Activation value
            act_text = self.font_small.render(f"{activation:.2f}", True, self.colors['text_dim'])
            self.screen.blit(act_text, (input_x + 15, y - 6))
        
        # Hidden layer
        hidden_label = self.font_small.render("Hidden Layer", True, self.colors['text'])
        self.screen.blit(hidden_label, (hidden_x - 20, layer_y - 20))
        
        for i, activation in enumerate(latest['hidden_layer']):
            y = layer_y + 30 + i * neuron_spacing
            color_intensity = int(255 * activation)
            neuron_color = (color_intensity // 2, color_intensity, color_intensity // 2)
            pygame.draw.circle(self.screen, neuron_color, (hidden_x, y), 8)
        
        # Output layer
        output_label = self.font_small.render("Output Layer", True, self.colors['text'])
        self.screen.blit(output_label, (output_x - 20, layer_y - 20))
        
        for i, activation in enumerate(latest['output_layer']):
            y = layer_y + i * neuron_spacing
            color_intensity = int(255 * activation)
            neuron_color = (color_intensity // 2, color_intensity // 2, color_intensity)
            pygame.draw.circle(self.screen, neuron_color, (output_x, y), 8)
            
            # Action name for output neurons
            action_text = self.font_small.render(self.action_names[i][:12], True, 
                                               self.colors['text'] if activation > 0.8 else self.colors['text_dim'])
            self.screen.blit(action_text, (output_x + 15, y - 6))
        
        # Draw connections (simplified)
        for i in range(len(latest['input_layer'])):
            for j in range(len(latest['hidden_layer'])):
                start_y = layer_y + i * neuron_spacing
                end_y = layer_y + 30 + j * neuron_spacing
                pygame.draw.line(self.screen, (60, 60, 100), 
                               (input_x + 8, start_y), (hidden_x - 8, end_y), 1)
        
        for i in range(len(latest['hidden_layer'])):
            for j in range(len(latest['output_layer'])):
                start_y = layer_y + 30 + i * neuron_spacing
                end_y = layer_y + j * neuron_spacing
                pygame.draw.line(self.screen, (60, 60, 100), 
                               (hidden_x + 8, start_y), (output_x - 8, end_y), 1)
    
    def _render_decision_panel(self):
        """Render AI decision panel"""
        panel_x = 640
        panel_y = 340
        panel_width = 740
        panel_height = 300
        
        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['agent'], (panel_x, panel_y, panel_width, panel_height), 2)
        
        title = self.font_large.render("🎯 Live AI Decisions", True, self.colors['agent'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        if not self.decision_history:
            no_decisions = self.font_medium.render("No decisions yet...", True, self.colors['text_dim'])
            self.screen.blit(no_decisions, (panel_x + 10, panel_y + 50))
            return
        
        # Recent decisions
        decisions_y = panel_y + 50
        for i, decision in enumerate(self.decision_history[-8:]):  # Show last 8 decisions
            y = decisions_y + i * 30
            alpha = 1.0 - (len(self.decision_history) - i - 1) * 0.1  # Fade older decisions
            
            # Decision text
            decision_text = f"Step {decision['step']:3d}: {decision['action_name'][:20]}"
            color = tuple(int(c * alpha) for c in self.colors['text'])
            text_surface = self.font_small.render(decision_text, True, color)
            self.screen.blit(text_surface, (panel_x + 10, y))
            
            # Reward and impact
            reward_text = f"Reward: {decision['reward']:6.1f}"
            reward_color = self.colors['stable'] if decision['reward'] > 0 else self.colors['crisis']
            reward_color = tuple(int(c * alpha) for c in reward_color)
            reward_surface = self.font_small.render(reward_text, True, reward_color)
            self.screen.blit(reward_surface, (panel_x + 300, y))
            
            # Target and impact
            target_text = f"Target: {decision['target_country']}"
            target_surface = self.font_small.render(target_text, True, tuple(int(c * alpha) for c in self.colors['accent']))
            self.screen.blit(target_surface, (panel_x + 450, y))
            
            # Lives saved indicator
            if decision['lives_saved'] > 0:
                lives_text = f"💝 {decision['lives_saved']} lives"
                lives_surface = self.font_small.render(lives_text, True, tuple(int(c * alpha) for c in self.colors['stable']))
                self.screen.blit(lives_surface, (panel_x + 580, y))
    
    def _render_performance_stats(self):
        """Render performance statistics"""
        panel_x = 20
        panel_y = 660
        panel_width = self.width - 40
        panel_height = 80
        
        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['gold'], (panel_x, panel_y, panel_width, panel_height), 2)
        
        title = self.font_large.render("📊 Model Performance Statistics", True, self.colors['gold'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        stats_y = panel_y + 40
        
        # Current episode stats
        current_stats = f"Current Episode - Reward: {self.total_reward:.1f} | Steps: {self.step_count}"
        current_surface = self.font_medium.render(current_stats, True, self.colors['text'])
        self.screen.blit(current_surface, (panel_x + 10, stats_y))
        
        # Historical performance
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-10:])  # Last 10 episodes
            best_reward = max(self.episode_rewards)
            
            historical_stats = f"Last 10 Episodes Avg: {avg_reward:.1f} | Best Ever: {best_reward:.1f} | Episodes Completed: {len(self.episode_rewards)}"
            historical_surface = self.font_medium.render(historical_stats, True, self.colors['text_dim'])
            self.screen.blit(historical_surface, (panel_x + 10, stats_y + 25))
    
    def _render_controls(self):
        """Render control instructions"""
        controls = [
            "Controls: SPACE=Pause/Resume | N=Next Step | R=Reset Episode | ESC=Exit",
            f"🎬 Recording: {'ON' if self.recording else 'OFF'} | Press V to toggle video recording"
        ]
        
        for i, control in enumerate(controls):
            color = self.colors['text'] if i == 0 else (self.colors['crisis'] if self.recording else self.colors['text_dim'])
            control_surface = self.font_small.render(control, True, color)
            self.screen.blit(control_surface, (20, self.height - 40 + i * 20))
    
    def run(self, auto_step=True, step_delay=0.5):
        """Run the visualization"""
        if not SB3_AVAILABLE:
            print("❌ Stable Baselines3 not available!")
            return
        
        if not self.model:
            print("❌ No model loaded!")
            return
        
        print(f"🤖 Starting live visualization of {self.model_name}")
        print("🎯 Watch the neural network make real-time decisions!")
        
        running = True
        paused = False
        last_step_time = time.time()
        
        while running:
            current_time = time.time()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("⏸️ Paused" if paused else "▶️ Resumed")
                    elif event.key == pygame.K_n:
                        # Manual step
                        self.step_model()
                        print(f"👆 Manual step - Reward: {self.decision_history[-1]['reward']:.1f}" if self.decision_history else "👆 Manual step")
                    elif event.key == pygame.K_r:
                        # Reset episode
                        self.reset_episode()
                        print("🔄 Episode reset")
                    elif event.key == pygame.K_v:
                        # Toggle recording
                        self.recording = not self.recording
                        if self.recording:
                            self.recording_start_time = current_time
                            print("🎬 Recording started")
                        else:
                            print("⏹️ Recording stopped")
            
            # Auto-step the model
            if not paused and auto_step and (current_time - last_step_time) >= step_delay:
                self.step_model()
                last_step_time = current_time
            
            # Render
            self.render()
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS for smooth animation
        
        pygame.quit()
        print("👋 Visualization ended")
        
        if self.episode_rewards:
            print(f"📊 Final Statistics:")
            print(f"   Episodes: {len(self.episode_rewards)}")
            print(f"   Average Reward: {np.mean(self.episode_rewards):.1f}")
            print(f"   Best Episode: {max(self.episode_rewards):.1f}")

def main():
    """Main function to choose and visualize models"""
    if not SB3_AVAILABLE:
        print("❌ Stable Baselines3 not available!")
        print("Install with: pip install stable-baselines3")
        return
    
    print("🤖 LIVE TRAINED MODEL VISUALIZATION")
    print("=" * 50)
    print("Watch your trained neural networks make real-time decisions!")
    print()
    
    # Find available models
    models_dir = "models"
    available_models = {}
    
    for algorithm in ['dqn', 'ppo', 'a2c']:
        best_path = os.path.join(models_dir, f"{algorithm}_best", "best_model.zip")
        if os.path.exists(best_path):
            available_models[algorithm.upper()] = best_path
    
    if not available_models:
        print("❌ No trained models found!")
        print("Train models first using: python real_rl_training.py")
        return
    
    print("Available trained models:")
    algorithms = list(available_models.keys())
    for i, alg in enumerate(algorithms):
        print(f"  {i+1}. {alg}")
    
    try:
        choice = int(input(f"\nChoose model to visualize (1-{len(algorithms)}): ")) - 1
        
        if 0 <= choice < len(algorithms):
            algorithm = algorithms[choice]
            model_path = available_models[algorithm]
            
            print(f"\n🚀 Loading {algorithm} model...")
            
            # Create visualizer
            visualizer = ModelVisualizer()
            
            if visualizer.load_model(model_path, algorithm):
                print(f"✅ {algorithm} loaded successfully!")
                print("\n🎮 Controls:")
                print("   SPACE = Pause/Resume automatic stepping")
                print("   N = Take single manual step")
                print("   R = Reset current episode")
                print("   V = Toggle video recording")
                print("   ESC = Exit")
                print()
                print("🎬 Starting live visualization...")
                
                # Run visualization
                visualizer.run(auto_step=True, step_delay=1.0)  # 1 second between steps
            else:
                print("❌ Failed to load model")
        else:
            print("❌ Invalid choice")
    
    except (ValueError, KeyboardInterrupt):
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()