"""
Crisis Response Environment - A Sophisticated RL Environment for Conflict Prediction

This environment simulates a complex geopolitical scenario where an AI agent must:
1. Monitor multiple regions for crisis indicators
2. Predict and prevent conflicts before they escalate
3. Manage limited resources efficiently
4. Issue timely alerts and recommendations
5. Coordinate international response efforts

Features:
- Rich state space with 50+ features per region
- Complex action space with discrete and continuous components
- Dynamic reward structure balancing multiple objectives
- Real-time visualization with advanced graphics
- Realistic simulation of conflict escalation patterns
"""

import gym
from gym import spaces
import numpy as np
import pygame
import random
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import math
import json
from dataclasses import dataclass
from collections import deque

class ConflictType(Enum):
    """Types of conflicts that can occur"""
    POLITICAL = "political"
    ECONOMIC = "economic"
    HUMANITARIAN = "humanitarian"
    ENVIRONMENTAL = "environmental"
    ETHNIC = "ethnic"
    TERRITORIAL = "territorial"

class AlertLevel(Enum):
    """Alert levels for crisis severity"""
    GREEN = 0      # No threat
    YELLOW = 1     # Low risk
    ORANGE = 2     # Medium risk
    RED = 3        # High risk
    CRITICAL = 4   # Imminent crisis

@dataclass
class Region:
    """Represents a geographical region with conflict indicators"""
    id: int
    name: str
    population: int
    economic_stability: float  # 0-1
    political_stability: float  # 0-1
    social_cohesion: float     # 0-1
    resource_scarcity: float   # 0-1
    external_pressure: float   # 0-1
    historical_conflicts: int
    current_alert_level: AlertLevel
    active_conflicts: List[ConflictType]
    resources_deployed: Dict[str, int]
    media_attention: float     # 0-1
    international_support: float  # 0-1
    climate_stress: float      # 0-1
    displacement_risk: float   # 0-1

class ActionType(Enum):
    """Available actions for the agent"""
    MONITOR = 0           # Increase monitoring in region
    DEPLOY_PEACEKEEPERS = 1  # Deploy peacekeeping forces
    ECONOMIC_AID = 2      # Provide economic assistance
    DIPLOMATIC_INTERVENTION = 3  # Diplomatic mediation
    HUMANITARIAN_AID = 4  # Emergency humanitarian assistance
    EARLY_WARNING = 5     # Issue early warning alert
    MEDIA_CAMPAIGN = 6    # Launch awareness campaign
    REFUGEE_SUPPORT = 7   # Support displaced populations
    WAIT = 8             # Take no action (observe)

class CrisisResponseEnv(gym.Env):
    """
    Advanced Crisis Response Environment for Reinforcement Learning
    
    State Space (300 dimensions):
    - Per region (12 regions × 25 features = 300):
      * Conflict indicators (10 features)
      * Resource allocation (5 features)
      * Historical data (5 features)
      * Environmental factors (5 features)
    
    Action Space (Discrete):
    - 9 action types × 12 regions = 108 possible actions
    - Plus intensity levels for certain actions
    
    Reward Structure:
    - Crisis prevention: +100 points
    - Early detection: +50 points
    - Efficient resource use: +20 points
    - False alarms: -10 points
    - Crisis escalation: -200 points
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 num_regions: int = 12,
                 max_episodes: int = 1000,
                 time_horizon: int = 100,
                 render_mode: Optional[str] = None):
        super(CrisisResponseEnv, self).__init__()
        
        # Environment configuration
        self.num_regions = num_regions
        self.max_episodes = max_episodes
        self.time_horizon = time_horizon
        self.render_mode = render_mode
        
        # Initialize regions with realistic data
        self.regions = self._initialize_regions()
        
        # Action space: [Action Type, Region ID, Intensity (0-1)]
        self.action_space = spaces.MultiDiscrete([
            len(ActionType),      # Action type
            self.num_regions,     # Target region
            11                    # Intensity level (0-10, mapped to 0-1)
        ])
        
        # State space: Rich multi-dimensional observation
        # Each region has 25 features, total 300 features
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.num_regions * 25,),
            dtype=np.float32
        )
        
        # Environment state
        self.current_step = 0
        self.episode_reward = 0
        self.crisis_count = 0
        self.prevented_crises = 0
        self.false_alarms = 0
        
        # Resource constraints
        self.total_budget = 1000.0
        self.remaining_budget = self.total_budget
        self.peacekeepers_available = 50
        self.diplomatic_capacity = 10
        
        # Historical tracking
        self.action_history = deque(maxlen=50)
        self.crisis_history = deque(maxlen=100)
        self.performance_metrics = {
            'episodes_completed': 0,
            'total_crises_prevented': 0,
            'total_false_alarms': 0,
            'average_response_time': 0,
            'resource_efficiency': 0
        }
        
        # Pygame initialization for visualization
        self.screen = None
        self.screen_width = 1200
        self.screen_height = 800
        self.clock = None
        
        # Colors for visualization
        self.colors = {
            'background': (20, 30, 40),
            'region_safe': (50, 150, 50),
            'region_warning': (200, 200, 50),
            'region_danger': (200, 50, 50),
            'region_critical': (150, 0, 0),
            'text': (255, 255, 255),
            'grid': (100, 100, 100),
            'agent': (50, 150, 255),
            'resource': (255, 200, 50)
        }
        
    def _initialize_regions(self) -> List[Region]:
        """Initialize regions with realistic geopolitical data"""
        region_names = [
            "Central Africa", "East Africa", "West Africa", "North Africa",
            "Middle East", "Central Asia", "South Asia", "Southeast Asia",
            "Eastern Europe", "Latin America", "Caribbean", "Pacific Islands"
        ]
        
        regions = []
        for i, name in enumerate(region_names):
            # Generate realistic initial conditions with some randomness
            base_stability = random.uniform(0.3, 0.8)
            
            region = Region(
                id=i,
                name=name,
                population=random.randint(5_000_000, 100_000_000),
                economic_stability=base_stability + random.uniform(-0.2, 0.2),
                political_stability=base_stability + random.uniform(-0.3, 0.2),
                social_cohesion=base_stability + random.uniform(-0.2, 0.3),
                resource_scarcity=random.uniform(0.2, 0.8),
                external_pressure=random.uniform(0.1, 0.6),
                historical_conflicts=random.randint(0, 5),
                current_alert_level=AlertLevel.GREEN,
                active_conflicts=[],
                resources_deployed={'peacekeepers': 0, 'aid': 0, 'monitors': 0},
                media_attention=random.uniform(0.1, 0.4),
                international_support=random.uniform(0.3, 0.7),
                climate_stress=random.uniform(0.2, 0.7),
                displacement_risk=random.uniform(0.1, 0.5)
            )
            
            # Ensure values are within valid range
            region.economic_stability = np.clip(region.economic_stability, 0, 1)
            region.political_stability = np.clip(region.political_stability, 0, 1)
            region.social_cohesion = np.clip(region.social_cohesion, 0, 1)
            
            regions.append(region)
            
        return regions
    
    def _get_observation(self) -> np.ndarray:
        """Generate rich observation vector from current state"""
        obs = []
        
        for region in self.regions:
            # Core stability indicators (5 features)
            obs.extend([
                region.economic_stability,
                region.political_stability,
                region.social_cohesion,
                region.resource_scarcity,
                region.external_pressure
            ])
            
            # Environmental and social factors (5 features)
            obs.extend([
                region.media_attention,
                region.international_support,
                region.climate_stress,
                region.displacement_risk,
                float(region.current_alert_level.value) / 4.0  # Normalized
            ])
            
            # Historical context (3 features)
            obs.extend([
                min(region.historical_conflicts / 10.0, 1.0),  # Normalized
                len(region.active_conflicts) / len(ConflictType),  # Normalized
                float(len([c for c in self.crisis_history if c['region_id'] == region.id])) / 10.0
            ])
            
            # Resource deployment (5 features)
            obs.extend([
                region.resources_deployed['peacekeepers'] / 50.0,  # Normalized
                region.resources_deployed['aid'] / 100.0,
                region.resources_deployed['monitors'] / 20.0,
                self.remaining_budget / self.total_budget,
                float(self.peacekeepers_available) / 50.0
            ])
            
            # Temporal features (4 features)
            obs.extend([
                float(self.current_step) / self.time_horizon,
                math.sin(2 * math.pi * self.current_step / 20),  # Seasonal pattern
                float(len(self.action_history)) / 50.0,
                random.uniform(0, 0.1)  # Noise/uncertainty
            ])
            
            # Conflict risk indicators (3 features)
            risk_factors = [
                1.0 - region.economic_stability,
                1.0 - region.political_stability,
                region.resource_scarcity
            ]
            obs.extend(risk_factors)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_conflict_probability(self, region: Region) -> float:
        """Calculate probability of conflict in a region based on various factors"""
        # Base risk from instability
        instability_risk = (
            (1.0 - region.economic_stability) * 0.3 +
            (1.0 - region.political_stability) * 0.4 +
            (1.0 - region.social_cohesion) * 0.3
        )
        
        # External factors
        external_risk = (
            region.resource_scarcity * 0.2 +
            region.external_pressure * 0.2 +
            region.climate_stress * 0.1 +
            region.displacement_risk * 0.15
        )
        
        # Historical factor
        historical_risk = min(region.historical_conflicts / 10.0, 0.2)
        
        # Current conflict escalation
        escalation_risk = len(region.active_conflicts) * 0.1
        
        # Protective factors
        protection = (
            region.international_support * 0.1 +
            (region.resources_deployed['peacekeepers'] / 50.0) * 0.1 +
            (region.resources_deployed['aid'] / 100.0) * 0.05
        )
        
        total_risk = instability_risk + external_risk + historical_risk + escalation_risk - protection
        
        # Add some randomness to simulate uncertainty
        total_risk += random.uniform(-0.05, 0.05)
        
        return np.clip(total_risk, 0.0, 1.0)
    
    def _update_alert_levels(self):
        """Update alert levels based on conflict probabilities"""
        for region in self.regions:
            prob = self._calculate_conflict_probability(region)
            
            if prob < 0.2:
                region.current_alert_level = AlertLevel.GREEN
            elif prob < 0.4:
                region.current_alert_level = AlertLevel.YELLOW
            elif prob < 0.6:
                region.current_alert_level = AlertLevel.ORANGE
            elif prob < 0.8:
                region.current_alert_level = AlertLevel.RED
            else:
                region.current_alert_level = AlertLevel.CRITICAL
    
    def _execute_action(self, action: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Execute the selected action and return reward and info"""
        action_type = ActionType(action[0])
        region_id = action[1]
        intensity = action[2] / 10.0  # Convert to 0-1 range
        
        if region_id >= len(self.regions):
            return -5.0, {'error': 'Invalid region ID'}
        
        region = self.regions[region_id]
        reward = 0.0
        info = {'action_type': action_type.name, 'region': region.name, 'intensity': intensity}
        
        # Calculate action costs
        action_cost = self._calculate_action_cost(action_type, intensity)
        
        if action_cost > self.remaining_budget:
            return -10.0, {'error': 'Insufficient budget', **info}
        
        # Execute specific actions
        if action_type == ActionType.MONITOR:
            region.resources_deployed['monitors'] += int(intensity * 5)
            region.media_attention = min(1.0, region.media_attention + intensity * 0.1)
            reward += 5.0 * intensity
            
        elif action_type == ActionType.DEPLOY_PEACEKEEPERS:
            if self.peacekeepers_available >= intensity * 10:
                deployed = int(intensity * 10)
                region.resources_deployed['peacekeepers'] += deployed
                self.peacekeepers_available -= deployed
                region.political_stability = min(1.0, region.political_stability + intensity * 0.1)
                reward += 15.0 * intensity
            else:
                return -5.0, {'error': 'Insufficient peacekeepers', **info}
                
        elif action_type == ActionType.ECONOMIC_AID:
            region.resources_deployed['aid'] += int(intensity * 20)
            region.economic_stability = min(1.0, region.economic_stability + intensity * 0.15)
            region.resource_scarcity = max(0.0, region.resource_scarcity - intensity * 0.1)
            reward += 10.0 * intensity
            
        elif action_type == ActionType.DIPLOMATIC_INTERVENTION:
            if self.diplomatic_capacity >= intensity * 2:
                self.diplomatic_capacity -= int(intensity * 2)
                region.external_pressure = max(0.0, region.external_pressure - intensity * 0.2)
                region.political_stability = min(1.0, region.political_stability + intensity * 0.1)
                reward += 20.0 * intensity
            else:
                return -5.0, {'error': 'Insufficient diplomatic capacity', **info}
                
        elif action_type == ActionType.HUMANITARIAN_AID:
            region.social_cohesion = min(1.0, region.social_cohesion + intensity * 0.1)
            region.displacement_risk = max(0.0, region.displacement_risk - intensity * 0.15)
            reward += 12.0 * intensity
            
        elif action_type == ActionType.EARLY_WARNING:
            # Issue early warning - effectiveness depends on timing
            conflict_prob = self._calculate_conflict_probability(region)
            if conflict_prob > 0.5:  # Justified warning
                region.international_support = min(1.0, region.international_support + intensity * 0.1)
                reward += 25.0 * intensity
            else:  # False alarm
                self.false_alarms += 1
                reward -= 10.0
                
        elif action_type == ActionType.MEDIA_CAMPAIGN:
            region.media_attention = min(1.0, region.media_attention + intensity * 0.2)
            region.international_support = min(1.0, region.international_support + intensity * 0.05)
            reward += 8.0 * intensity
            
        elif action_type == ActionType.REFUGEE_SUPPORT:
            region.displacement_risk = max(0.0, region.displacement_risk - intensity * 0.2)
            region.social_cohesion = min(1.0, region.social_cohesion + intensity * 0.05)
            reward += 15.0 * intensity
            
        elif action_type == ActionType.WAIT:
            # Waiting has low cost but provides observation benefits
            reward += 1.0
            
        # Deduct action cost
        self.remaining_budget -= action_cost
        
        # Record action in history
        self.action_history.append({
            'step': self.current_step,
            'action_type': action_type.name,
            'region_id': region_id,
            'intensity': intensity,
            'reward': reward
        })
        
        return reward, info
    
    def _calculate_action_cost(self, action_type: ActionType, intensity: float) -> float:
        """Calculate the cost of an action based on type and intensity"""
        base_costs = {
            ActionType.MONITOR: 5.0,
            ActionType.DEPLOY_PEACEKEEPERS: 50.0,
            ActionType.ECONOMIC_AID: 30.0,
            ActionType.DIPLOMATIC_INTERVENTION: 20.0,
            ActionType.HUMANITARIAN_AID: 25.0,
            ActionType.EARLY_WARNING: 10.0,
            ActionType.MEDIA_CAMPAIGN: 15.0,
            ActionType.REFUGEE_SUPPORT: 35.0,
            ActionType.WAIT: 1.0
        }
        
        return base_costs[action_type] * intensity
    
    def _simulate_environment_dynamics(self):
        """Simulate natural evolution of the environment"""
        for region in self.regions:
            # Natural stability changes
            region.economic_stability += random.uniform(-0.02, 0.01)
            region.political_stability += random.uniform(-0.02, 0.01)
            region.social_cohesion += random.uniform(-0.01, 0.01)
            
            # External pressures
            region.external_pressure += random.uniform(-0.01, 0.02)
            region.climate_stress += random.uniform(-0.005, 0.01)
            
            # Resource depletion
            if region.resources_deployed['peacekeepers'] > 0:
                region.resources_deployed['peacekeepers'] = max(0, 
                    region.resources_deployed['peacekeepers'] - random.randint(0, 2))
            
            # Clamp values
            region.economic_stability = np.clip(region.economic_stability, 0, 1)
            region.political_stability = np.clip(region.political_stability, 0, 1)
            region.social_cohesion = np.clip(region.social_cohesion, 0, 1)
            region.external_pressure = np.clip(region.external_pressure, 0, 1)
            region.climate_stress = np.clip(region.climate_stress, 0, 1)
            
            # Check for conflict emergence
            conflict_prob = self._calculate_conflict_probability(region)
            if conflict_prob > 0.8 and random.random() < 0.1:
                # New conflict emerges
                if len(region.active_conflicts) < 2:
                    new_conflict = random.choice(list(ConflictType))
                    if new_conflict not in region.active_conflicts:
                        region.active_conflicts.append(new_conflict)
                        self.crisis_count += 1
                        self.crisis_history.append({
                            'step': self.current_step,
                            'region_id': region.id,
                            'conflict_type': new_conflict.name,
                            'probability': conflict_prob
                        })
    
    def _calculate_episode_reward(self) -> float:
        """Calculate additional rewards based on episode performance"""
        reward = 0.0
        
        # Crisis prevention bonus
        total_risk = sum(self._calculate_conflict_probability(region) for region in self.regions)
        prevention_bonus = max(0, (12.0 - total_risk) * 10)  # Bonus for keeping total risk low
        reward += prevention_bonus
        
        # Resource efficiency bonus
        efficiency = (self.total_budget - self.remaining_budget) / self.total_budget
        if efficiency < 0.8:  # Bonus for not overspending
            reward += 20.0
        
        # Early warning accuracy
        if self.false_alarms < 3:
            reward += 30.0
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        # Execute action
        action_reward, action_info = self._execute_action(action)
        
        # Simulate environment dynamics
        self._simulate_environment_dynamics()
        
        # Update alert levels
        self._update_alert_levels()
        
        # Calculate total reward
        total_reward = action_reward
        
        # Update counters
        self.current_step += 1
        self.episode_reward += total_reward
        
        # Check if episode is done
        done = (self.current_step >= self.time_horizon or 
                self.remaining_budget <= 0 or
                self.crisis_count >= 10)
        
        # Add episode completion bonus
        if done:
            total_reward += self._calculate_episode_reward()
        
        # Get new observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'crisis_count': self.crisis_count,
            'prevented_crises': self.prevented_crises,
            'false_alarms': self.false_alarms,
            'remaining_budget': self.remaining_budget,
            'peacekeepers_available': self.peacekeepers_available,
            'total_risk': sum(self._calculate_conflict_probability(region) for region in self.regions),
            **action_info
        }
        
        return observation, total_reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        # Reset environment state
        self.current_step = 0
        self.episode_reward = 0
        self.crisis_count = 0
        self.prevented_crises = 0
        self.false_alarms = 0
        
        # Reset resources
        self.remaining_budget = self.total_budget
        self.peacekeepers_available = 50
        self.diplomatic_capacity = 10
        
        # Reset regions
        self.regions = self._initialize_regions()
        
        # Clear histories
        self.action_history.clear()
        self.crisis_history.clear()
        
        # Update performance metrics
        self.performance_metrics['episodes_completed'] += 1
        
        return self._get_observation()
    
    def render(self, mode='human'):
        """Render the environment visualization"""
        if self.screen is None and mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Crisis Response AI - Reinforcement Learning Environment")
            self.clock = pygame.time.Clock()
        
        if mode == 'human':
            self._render_pygame()
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
    
    def _render_pygame(self):
        """Render the environment using Pygame"""
        if self.screen is None:
            return
            
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Calculate grid layout for regions (4x3 grid)
        cols, rows = 4, 3
        cell_width = self.screen_width // cols
        cell_height = (self.screen_height - 100) // rows  # Leave space for info panel
        
        # Render regions
        for i, region in enumerate(self.regions):
            row = i // cols
            col = i % cols
            x = col * cell_width
            y = row * cell_height
            
            # Determine region color based on alert level
            alert_colors = {
                AlertLevel.GREEN: self.colors['region_safe'],
                AlertLevel.YELLOW: (150, 150, 50),
                AlertLevel.ORANGE: self.colors['region_warning'],
                AlertLevel.RED: self.colors['region_danger'],
                AlertLevel.CRITICAL: self.colors['region_critical']
            }
            
            color = alert_colors.get(region.current_alert_level, self.colors['region_safe'])
            
            # Draw region rectangle
            pygame.draw.rect(self.screen, color, (x + 5, y + 5, cell_width - 10, cell_height - 10))
            pygame.draw.rect(self.screen, self.colors['grid'], 
                           (x + 5, y + 5, cell_width - 10, cell_height - 10), 2)
            
            # Region name
            font = pygame.font.Font(None, 24)
            text = font.render(region.name, True, self.colors['text'])
            text_rect = text.get_rect(center=(x + cell_width // 2, y + 20))
            self.screen.blit(text, text_rect)
            
            # Alert level
            alert_text = font.render(f"Alert: {region.current_alert_level.name}", True, self.colors['text'])
            alert_rect = alert_text.get_rect(center=(x + cell_width // 2, y + 45))
            self.screen.blit(alert_text, alert_rect)
            
            # Stability indicators
            small_font = pygame.font.Font(None, 18)
            
            stability_info = [
                f"Economic: {region.economic_stability:.2f}",
                f"Political: {region.political_stability:.2f}",
                f"Social: {region.social_cohesion:.2f}",
                f"Resources: {region.resource_scarcity:.2f}"
            ]
            
            for j, info in enumerate(stability_info):
                info_text = small_font.render(info, True, self.colors['text'])
                self.screen.blit(info_text, (x + 10, y + 70 + j * 20))
            
            # Show deployed resources
            if region.resources_deployed['peacekeepers'] > 0:
                pk_text = small_font.render(f"PK: {region.resources_deployed['peacekeepers']}", 
                                          True, (100, 200, 255))
                self.screen.blit(pk_text, (x + 10, y + cell_height - 40))
            
            if region.resources_deployed['aid'] > 0:
                aid_text = small_font.render(f"Aid: {region.resources_deployed['aid']}", 
                                           True, (255, 200, 100))
                self.screen.blit(aid_text, (x + 100, y + cell_height - 40))
        
        # Render information panel at bottom
        info_y = rows * cell_height + 10
        info_font = pygame.font.Font(None, 24)
        
        info_items = [
            f"Step: {self.current_step}/{self.time_horizon}",
            f"Budget: ${self.remaining_budget:.0f}/${self.total_budget:.0f}",
            f"Peacekeepers: {self.peacekeepers_available}/50",
            f"Crises: {self.crisis_count}",
            f"Episode Reward: {self.episode_reward:.1f}"
        ]
        
        for i, item in enumerate(info_items):
            text = info_font.render(item, True, self.colors['text'])
            self.screen.blit(text, (10 + i * 200, info_y))
        
        # Show recent actions
        if self.action_history:
            recent_action = self.action_history[-1]
            action_text = f"Last Action: {recent_action['action_type']} (Region {recent_action['region_id']})"
            action_surface = info_font.render(action_text, True, (255, 255, 100))
            self.screen.blit(action_surface, (10, info_y + 30))
    
    def close(self):
        """Clean up the environment"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics"""
        if self.performance_metrics['episodes_completed'] == 0:
            return self.performance_metrics
            
        total_episodes = self.performance_metrics['episodes_completed']
        
        # Calculate average response time
        if self.action_history:
            response_times = []
            for crisis in self.crisis_history:
                crisis_step = crisis['step']
                region_id = crisis['region_id']
                
                # Find first action in response to this crisis
                for action in self.action_history:
                    if (action['step'] >= crisis_step and 
                        action['region_id'] == region_id and
                        action['action_type'] != 'WAIT'):
                        response_time = action['step'] - crisis_step
                        response_times.append(response_time)
                        break
            
            if response_times:
                self.performance_metrics['average_response_time'] = np.mean(response_times)
        
        # Calculate resource efficiency
        total_spent = self.total_budget - self.remaining_budget
        if total_spent > 0:
            efficiency = (self.prevented_crises + 1) / (total_spent / 100)  # Crises prevented per 100 budget units
            self.performance_metrics['resource_efficiency'] = efficiency
        
        return self.performance_metrics
    
    def save_episode_data(self, filename: str):
        """Save episode data for analysis"""
        data = {
            'episode_metrics': self.get_performance_metrics(),
            'final_state': {
                'regions': [
                    {
                        'name': region.name,
                        'alert_level': region.current_alert_level.name,
                        'economic_stability': region.economic_stability,
                        'political_stability': region.political_stability,
                        'social_cohesion': region.social_cohesion,
                        'active_conflicts': [c.name for c in region.active_conflicts],
                        'resources_deployed': region.resources_deployed
                    }
                    for region in self.regions
                ],
                'remaining_budget': self.remaining_budget,
                'crisis_count': self.crisis_count,
                'episode_reward': self.episode_reward
            },
            'action_history': list(self.action_history),
            'crisis_history': list(self.crisis_history)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


def make_crisis_env(**kwargs):
    """Factory function to create CrisisResponseEnv"""
    return CrisisResponseEnv(**kwargs)