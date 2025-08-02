"""
Interactive Dashboard for Crisis Response AI

This module provides sophisticated dashboard components for:
- Real-time performance monitoring
- Training progress visualization
- Algorithm comparison displays
- Interactive control panels
- Comprehensive metrics analysis

Features:
- Live updating charts and graphs
- Interactive parameter controls
- Multi-algorithm comparison views
- Performance analytics
- Professional styling
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import threading
import time

class MetricsPanel:
    """Panel for displaying performance metrics and statistics"""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # Metrics tracking
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.success_rates = deque(maxlen=100)
        self.loss_values = deque(maxlen=1000)
        
        # Colors
        self.colors = {
            'background': (25, 35, 45),
            'border': (70, 80, 90),
            'text': (255, 255, 255),
            'accent': (100, 200, 255),
            'success': (100, 255, 100),
            'warning': (255, 200, 100),
            'danger': (255, 100, 100)
        }
        
        # Fonts
        pygame.font.init()
        self.fonts = {
            'title': pygame.font.Font(None, 24),
            'normal': pygame.font.Font(None, 18),
            'small': pygame.font.Font(None, 14)
        }
        
    def update_metrics(self, episode_reward: float, episode_length: int, 
                      success_rate: float = None, loss_value: float = None):
        """Update metrics with new data"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        if success_rate is not None:
            self.success_rates.append(success_rate)
        if loss_value is not None:
            self.loss_values.append(loss_value)
    
    def render(self, screen: pygame.Surface):
        """Render the metrics panel"""
        # Background
        pygame.draw.rect(screen, self.colors['background'], 
                        (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, self.colors['border'], 
                        (self.x, self.y, self.width, self.height), 2)
        
        # Title
        title = self.fonts['title'].render("Performance Metrics", True, self.colors['text'])
        screen.blit(title, (self.x + 10, self.y + 10))
        
        current_y = self.y + 40
        
        # Current metrics
        if self.episode_rewards:
            metrics = [
                ("Latest Reward", f"{self.episode_rewards[-1]:.1f}"),
                ("Average Reward", f"{np.mean(self.episode_rewards):.1f}"),
                ("Best Reward", f"{np.max(self.episode_rewards):.1f}"),
                ("Episodes Completed", str(len(self.episode_rewards))),
            ]
            
            if self.success_rates:
                metrics.append(("Success Rate", f"{self.success_rates[-1]:.1%}"))
            
            for label, value in metrics:
                label_surface = self.fonts['normal'].render(label, True, self.colors['text'])
                value_surface = self.fonts['normal'].render(value, True, self.colors['accent'])
                
                screen.blit(label_surface, (self.x + 10, current_y))
                value_rect = value_surface.get_rect()
                value_rect.right = self.x + self.width - 10
                value_rect.y = current_y
                screen.blit(value_surface, value_rect)
                
                current_y += 25
        
        # Mini charts
        if len(self.episode_rewards) > 1:
            self._render_mini_chart(screen, self.episode_rewards, 
                                  self.x + 10, current_y + 20, 
                                  self.width - 20, 60, "Episode Rewards")

class CrisisDashboard:
    """Main dashboard for the Crisis Response AI system"""
    
    def __init__(self, width: int = 1600, height: int = 1000):
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Crisis Response AI - Control Dashboard")
        
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Colors and styling
        self.colors = {
            'bg_primary': (15, 25, 35),
            'bg_secondary': (25, 35, 45),
            'bg_tertiary': (35, 45, 55),
            'border': (70, 80, 90),
            'text': (255, 255, 255),
            'text_secondary': (200, 200, 200),
            'accent': (100, 200, 255),
            'success': (100, 255, 100),
            'warning': (255, 200, 100),
            'danger': (255, 100, 100)
        }
        
        # Layout configuration
        self.layout = {
            'header_height': 80,
            'sidebar_width': 300,
            'bottom_panel_height': 200,
            'panel_margin': 10
        }
        
        # Dashboard components
        self.metrics_panel = MetricsPanel(
            self.layout['sidebar_width'] + self.layout['panel_margin'],
            self.layout['header_height'] + self.layout['panel_margin'],
            self.width - self.layout['sidebar_width'] - 2 * self.layout['panel_margin'],
            200
        )
        
        # Training control state
        self.training_state = {
            'is_training': False,
            'is_paused': False,
            'current_algorithm': 'DQN',
            'current_episode': 0,
            'total_episodes': 1000
        }
        
        # Algorithm comparison data
        self.algorithm_data = {
            'DQN': {'rewards': deque(maxlen=1000), 'color': (255, 100, 100)},
            'PPO': {'rewards': deque(maxlen=1000), 'color': (100, 255, 100)},
            'REINFORCE': {'rewards': deque(maxlen=1000), 'color': (100, 100, 255)},
            'A2C': {'rewards': deque(maxlen=1000), 'color': (255, 255, 100)}
        }
        
    def render_header(self):
        """Render dashboard header"""
        # Background
        pygame.draw.rect(self.screen, self.colors['bg_secondary'], 
                        (0, 0, self.width, self.layout['header_height']))
        pygame.draw.line(self.screen, self.colors['border'], 
                        (0, self.layout['header_height']), 
                        (self.width, self.layout['header_height']), 2)
        
        # Title
        font_large = pygame.font.Font(None, 36)
        title = font_large.render("Crisis Response AI - Training Dashboard", 
                                True, self.colors['text'])
        self.screen.blit(title, (20, 20))
        
        # Training status
        font_medium = pygame.font.Font(None, 24)
        status_text = "TRAINING" if self.training_state['is_training'] else "IDLE"
        status_color = self.colors['success'] if self.training_state['is_training'] else self.colors['text_secondary']
        status = font_medium.render(f"Status: {status_text}", True, status_color)
        self.screen.blit(status, (20, 50))
        
        # Current algorithm
        algo_text = f"Algorithm: {self.training_state['current_algorithm']}"
        algo = font_medium.render(algo_text, True, self.colors['accent'])
        self.screen.blit(algo, (200, 50))
        
        # Progress
        episode_text = f"Episode: {self.training_state['current_episode']}/{self.training_state['total_episodes']}"
        episode = font_medium.render(episode_text, True, self.colors['text'])
        self.screen.blit(episode, (400, 50))
    
    def render_control_panel(self):
        """Render training control panel"""
        panel_x = 10
        panel_y = self.layout['header_height'] + 10
        panel_width = self.layout['sidebar_width'] - 20
        panel_height = self.height - self.layout['header_height'] - self.layout['bottom_panel_height'] - 30
        
        # Background
        pygame.draw.rect(self.screen, self.colors['bg_secondary'], 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['border'], 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        font_medium = pygame.font.Font(None, 20)
        title = font_medium.render("Training Controls", True, self.colors['text'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        current_y = panel_y + 40
        
        # Control buttons (simplified rendering)
        buttons = [
            "Start Training",
            "Pause/Resume",
            "Stop Training",
            "Save Model",
            "Load Model",
            "Reset Environment"
        ]
        
        font_small = pygame.font.Font(None, 16)
        for button in buttons:
            button_color = self.colors['bg_tertiary']
            if button == "Start Training" and not self.training_state['is_training']:
                button_color = self.colors['success']
            elif button == "Stop Training" and self.training_state['is_training']:
                button_color = self.colors['danger']
            
            pygame.draw.rect(self.screen, button_color, 
                           (panel_x + 10, current_y, panel_width - 20, 30))
            pygame.draw.rect(self.screen, self.colors['border'], 
                           (panel_x + 10, current_y, panel_width - 20, 30), 1)
            
            button_text = font_small.render(button, True, self.colors['text'])
            text_rect = button_text.get_rect(center=(panel_x + panel_width // 2, current_y + 15))
            self.screen.blit(button_text, text_rect)
            
            current_y += 40
        
        # Algorithm selection
        current_y += 20
        algo_title = font_medium.render("Select Algorithm:", True, self.colors['text'])
        self.screen.blit(algo_title, (panel_x + 10, current_y))
        current_y += 30
        
        algorithms = ['DQN', 'PPO', 'REINFORCE', 'A2C']
        for algo in algorithms:
            is_selected = algo == self.training_state['current_algorithm']
            color = self.colors['accent'] if is_selected else self.colors['text_secondary']
            
            algo_text = font_small.render(f"● {algo}", True, color)
            self.screen.blit(algo_text, (panel_x + 20, current_y))
            current_y += 25
    
    def render_comparison_chart(self):
        """Render algorithm comparison chart"""
        chart_x = self.layout['sidebar_width'] + 20
        chart_y = self.layout['header_height'] + 230
        chart_width = self.width - self.layout['sidebar_width'] - 40
        chart_height = 300
        
        # Background
        pygame.draw.rect(self.screen, self.colors['bg_secondary'], 
                        (chart_x, chart_y, chart_width, chart_height))
        pygame.draw.rect(self.screen, self.colors['border'], 
                        (chart_x, chart_y, chart_width, chart_height), 2)
        
        # Title
        font_medium = pygame.font.Font(None, 20)
        title = font_medium.render("Algorithm Performance Comparison", True, self.colors['text'])
        self.screen.blit(title, (chart_x + 10, chart_y + 10))
        
        # Render chart data
        if any(len(data['rewards']) > 1 for data in self.algorithm_data.values()):
            self._render_performance_chart(chart_x + 10, chart_y + 40, 
                                         chart_width - 20, chart_height - 50)
    
    def _render_performance_chart(self, x: int, y: int, width: int, height: int):
        """Render performance comparison chart"""
        # Simple line chart rendering
        max_points = 100
        
        # Find global min/max for scaling
        all_rewards = []
        for data in self.algorithm_data.values():
            if len(data['rewards']) > 0:
                all_rewards.extend(list(data['rewards'])[-max_points:])
        
        if not all_rewards:
            return
        
        min_reward = min(all_rewards)
        max_reward = max(all_rewards)
        reward_range = max_reward - min_reward if max_reward != min_reward else 1
        
        # Draw axes
        pygame.draw.line(self.screen, self.colors['border'], (x, y + height), (x + width, y + height), 1)
        pygame.draw.line(self.screen, self.colors['border'], (x, y), (x, y + height), 1)
        
        # Draw algorithm lines
        for algo_name, data in self.algorithm_data.items():
            rewards = list(data['rewards'])[-max_points:]
            if len(rewards) < 2:
                continue
            
            points = []
            for i, reward in enumerate(rewards):
                px = x + (i / max(len(rewards) - 1, 1)) * width
                py = y + height - ((reward - min_reward) / reward_range) * height
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, data['color'], False, points, 2)
        
        # Legend
        legend_x = x + width - 150
        legend_y = y + 10
        font_small = pygame.font.Font(None, 16)
        
        for i, (algo_name, data) in enumerate(self.algorithm_data.items()):
            if len(data['rewards']) > 0:
                # Color box
                pygame.draw.rect(self.screen, data['color'], 
                               (legend_x, legend_y + i * 20, 15, 15))
                # Label
                label = font_small.render(f"{algo_name} ({len(data['rewards'])})", 
                                        True, self.colors['text'])
                self.screen.blit(label, (legend_x + 20, legend_y + i * 20))
    
    def update_algorithm_data(self, algorithm: str, reward: float):
        """Update performance data for an algorithm"""
        if algorithm in self.algorithm_data:
            self.algorithm_data[algorithm]['rewards'].append(reward)
    
    def handle_events(self):
        """Handle pygame events"""
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.training_state['is_paused'] = not self.training_state['is_paused']
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle button clicks (simplified)
                self._handle_button_click(event.pos)
        
        return events
    
    def _handle_button_click(self, pos: Tuple[int, int]):
        """Handle button clicks in control panel"""
        # Simplified button handling
        # In a full implementation, this would check button boundaries
        # and execute corresponding actions
        pass
    
    def render_frame(self):
        """Render complete dashboard frame"""
        # Clear screen
        self.screen.fill(self.colors['bg_primary'])
        
        # Render components
        self.render_header()
        self.render_control_panel()
        self.metrics_panel.render(self.screen)
        self.render_comparison_chart()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
    
    def run_dashboard(self):
        """Run the dashboard main loop"""
        while self.running:
            self.handle_events()
            self.render_frame()
        
        pygame.quit()
    
    def cleanup(self):
        """Clean up dashboard resources"""
        pygame.quit()