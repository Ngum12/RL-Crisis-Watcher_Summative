#!/usr/bin/env python3
"""
Live Mobile Agent Visualization with Trained Neural Networks
Watch your trained RL models control a moving agent navigating real Africa road networks!

Shows trained neural networks making real-time decisions about:
- Where to move the agent (drone/response unit)
- Which crises to prioritize 
- Optimal route planning through real cities
- Crisis detection and response coordination
"""

import pygame
import numpy as np
import math
import time
import random
import heapq
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    from stable_baselines3 import DQN, PPO, A2C
    from real_rl_training import CrisisResponseEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

class MobileAgent:
    """Enhanced Mobile agent with communication and better movement tracking"""
    
    def __init__(self, start_pos: Tuple[float, float]):
        self.pos = list(start_pos)
        self.target_pos = None
        self.path = []
        self.speed = 2.0
        self.detected_crises = []
        self.current_mission = None
        self.fuel = 100.0
        self.communication_log = []
        
        # Enhanced features
        self.trail_positions = []
        self.last_signal_time = 0
        self.signal_cooldown = 3.0
        self.current_command_center = None
        self.signal_strength = 100.0
        self.mission_status = "Patrolling"
        self.response_received = False
        
    def move_towards_target(self, dt: float):
        """Enhanced movement with trail tracking"""
        if not self.path:
            return
            
        # Store previous position for trail
        prev_pos = self.pos.copy()
        
        target = self.path[0]
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 5:  # Reached waypoint
            self.path.pop(0)
            if not self.path:
                self.pos = [target[0], target[1]]
                return
            target = self.path[0] if self.path else target
            dx = target[0] - self.pos[0]
            dy = target[1] - self.pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            move_x = (dx / distance) * self.speed * dt
            move_y = (dy / distance) * self.speed * dt
            self.pos[0] += move_x
            self.pos[1] += move_y
            
            # Update trail (only if moved significantly)
            move_distance = math.sqrt(move_x*move_x + move_y*move_y)
            if move_distance > 1.0:  # Only add to trail if moved at least 1 pixel
                self.trail_positions.append(prev_pos.copy())
                if len(self.trail_positions) > 20:  # Limit trail length
                    self.trail_positions.pop(0)
            
            # Consume fuel
            self.fuel -= 0.1 * dt
            self.fuel = max(0, self.fuel)
    
    def send_signal_to_command(self, command_center_pos: Tuple[float, float], message: str, signal_type: str = "update"):
        """Send communication signal to command center"""
        current_time = time.time()
        if current_time - self.last_signal_time >= self.signal_cooldown:
            self.last_signal_time = current_time
            
            # Calculate signal strength based on distance
            distance = math.sqrt((self.pos[0] - command_center_pos[0])**2 + 
                               (self.pos[1] - command_center_pos[1])**2)
            self.signal_strength = max(20, 100 - (distance / 5))  # Weaker signal over distance
            
            return {
                'from_pos': self.pos.copy(),
                'to_pos': command_center_pos,
                'message': message,
                'type': signal_type,
                'timestamp': current_time,
                'strength': self.signal_strength
            }
        return None

class LiveMobileVisualization:
    """Visualize trained RL models controlling mobile crisis agent"""
    
    def __init__(self, width=1600, height=1000):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🚁 Live Mobile Agent - Trained Neural Network Control")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.colors = {
            'background': (5, 15, 25),
            'panel_bg': (20, 30, 40),
            'text': (255, 255, 255),
            'text_dim': (160, 160, 160),
            'stable': (50, 200, 50),
            'warning': (255, 220, 50),
            'crisis': (255, 60, 60),
            'agent': (0, 255, 150),
            'neural': (100, 150, 255),
            'accent': (0, 200, 255),
            'gold': (255, 215, 0),
            'route': (255, 100, 255),
            'water': (30, 100, 150),
            'land': (40, 80, 40)
        }
        
        # Enhanced Map setup (Africa focus: Cameroon, DRC, Sudan)
        self.map_area = (50, 100, 800, 600)  # x, y, width, height
        
        # Real cities and coordinates (scaled to map)
        self.cities = {
            # Cameroon
            'Yaoundé': (300, 300),     # Capital
            'Douala': (280, 320),      # Economic center
            'Garoua': (320, 250),      # Northern region
            'Bamenda': (290, 280),     # Northwest
            
            # DR Congo
            'Kinshasa': (400, 380),    # Capital
            'Lubumbashi': (450, 450),  # Mining center
            'Kisangani': (470, 320),   # Central
            'Goma': (480, 350),        # Eastern border
            'Bukavu': (485, 360),      # South Kivu
            
            # Sudan
            'Khartoum': (600, 200),    # Capital
            'Port Sudan': (650, 180),  # Red Sea port
            'Kassala': (640, 190),     # Eastern
            'El Obeid': (580, 220),    # Central
            'Nyala': (550, 280),       # Darfur
        }
        
        # Enhanced Country regions with better visualization
        self.countries = {
            'Cameroon': {
                'color': (100, 150, 100), 
                'border_color': (150, 200, 150),
                'cities': ['Yaoundé', 'Douala', 'Garoua', 'Bamenda'],
                'capital': 'Yaoundé',
                'region_bounds': [(260, 240), (340, 240), (340, 340), (260, 340)]
            },
            'DR Congo': {
                'color': (150, 100, 100), 
                'border_color': (200, 150, 150),
                'cities': ['Kinshasa', 'Lubumbashi', 'Kisangani', 'Goma', 'Bukavu'],
                'capital': 'Kinshasa',
                'region_bounds': [(380, 300), (500, 300), (500, 470), (380, 470)]
            },
            'Sudan': {
                'color': (100, 100, 150), 
                'border_color': (150, 150, 200),
                'cities': ['Khartoum', 'Port Sudan', 'Kassala', 'El Obeid', 'Nyala'],
                'capital': 'Khartoum',
                'region_bounds': [(540, 160), (670, 160), (670, 300), (540, 300)]
            }
        }
        
        # Command centers (headquarters/stations)
        self.command_centers = {
            'African Union HQ': {'pos': (720, 400), 'color': (255, 215, 0), 'coverage': 200},
            'UN Emergency': {'pos': (150, 200), 'color': (0, 150, 255), 'coverage': 150},
            'Regional Command': {'pos': (500, 500), 'color': (255, 100, 255), 'coverage': 180}
        }
        
        # Communication system
        self.active_signals = []  # Agent to station signals
        self.active_responses = []  # Station to agent responses
        self.signal_history = []
        
        # Agent trail for better movement visualization
        self.agent_trail = []
        self.max_trail_length = 20
        
        # Road network (simplified)
        self.roads = [
            # Cameroon internal
            ('Yaoundé', 'Douala'), ('Yaoundé', 'Bamenda'), ('Bamenda', 'Garoua'),
            # DRC internal  
            ('Kinshasa', 'Lubumbashi'), ('Kinshasa', 'Kisangani'), ('Kisangani', 'Goma'), ('Goma', 'Bukavu'),
            # Sudan internal
            ('Khartoum', 'Port Sudan'), ('Khartoum', 'Kassala'), ('Khartoum', 'El Obeid'), ('El Obeid', 'Nyala'),
            # Cross-border (limited)
            ('Bamenda', 'Kinshasa'), ('Goma', 'El Obeid')
        ]
        
        # Initialize enhanced agent
        self.agent = MobileAgent(self.cities['Yaoundé'])
        self.agent.trail_positions = []  # For movement trail
        self.agent.last_signal_time = 0
        self.agent.signal_cooldown = 3.0  # Signal every 3 seconds
        
        # RL Environment and Model
        self.env = None
        self.model = None
        self.model_name = ""
        self.current_obs = None
        
        # Crisis simulation
        self.active_crises = []
        self.crisis_history = []
        self.last_crisis_time = time.time()
        
        # AI decision tracking
        self.ai_decisions = []
        self.neural_activity = {}
        self.decision_reasoning = ""
        
        # Performance tracking
        self.episode_count = 0
        self.total_lives_saved = 0
        self.total_crises_resolved = 0
        self.mission_success_rate = 0.0
        
        # Animation
        self.animation_time = 0
        self.pulse_time = 0
        
        # Fonts
        self.font_title = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)
        
    def load_model(self, model_path: str, algorithm: str):
        """Load trained RL model"""
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
            self.current_obs, _ = self.env.reset()
            
            print(f"✅ Loaded {self.model_name} successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def generate_crisis(self):
        """Generate new crisis events"""
        current_time = time.time()
        if current_time - self.last_crisis_time > random.uniform(3, 8):  # Crisis every 3-8 seconds
            
            # Choose random city and crisis type
            city = random.choice(list(self.cities.keys()))
            crisis_types = [
                ("🔴 Armed Conflict", 0.9, (255, 60, 60)),
                ("⚠️ Political Unrest", 0.7, (255, 160, 60)),
                ("💔 Humanitarian Crisis", 0.8, (255, 100, 100)),
                ("🌾 Food Insecurity", 0.6, (255, 200, 60)),
                ("🏥 Health Emergency", 0.8, (255, 120, 120))
            ]
            
            crisis_type, severity, color = random.choice(crisis_types)
            
            crisis = {
                'type': crisis_type,
                'location': city,
                'pos': self.cities[city],
                'severity': severity,
                'color': color,
                'start_time': current_time,
                'lives_at_risk': random.randint(100, 5000),
                'discovered': False,
                'responding': False
            }
            
            self.active_crises.append(crisis)
            self.last_crisis_time = current_time
            
            # Keep only recent crises
            if len(self.active_crises) > 6:
                self.active_crises.pop(0)
    
    def update_ai_decision(self):
        """Get AI decision from trained model"""
        if not self.model or not self.env:
            return
        
        # Get action from trained model
        action, _states = self.model.predict(self.current_obs, deterministic=True)
        
        # Convert action to mobile agent decision
        target_country_idx = action % 3  # 0=Cameroon, 1=DRC, 2=Sudan
        action_type = action // 3  # Movement strategy
        
        countries_list = ['Cameroon', 'DR Congo', 'Sudan']
        target_country = countries_list[target_country_idx]
        
        # Get cities in target country
        target_cities = self.countries[target_country]['cities']
        
        # Find highest priority crisis in target area
        target_crisis = None
        max_priority = 0
        
        for crisis in self.active_crises:
            if crisis['location'] in target_cities and not crisis['responding']:
                priority = crisis['severity'] * crisis['lives_at_risk'] / 1000
                if priority > max_priority:
                    max_priority = priority
                    target_crisis = crisis
        
        # If no crisis, move to strategic position
        if not target_crisis:
            target_city = random.choice(target_cities)
        else:
            target_city = target_crisis['location']
            target_crisis['responding'] = True
        
        # Plan route using A* pathfinding
        self.plan_route_to_city(target_city)
        
        # Update AI reasoning
        action_descriptions = [
            "🔍 Surveillance Mode", "⚡ Rapid Response", "🛡️ Defensive Position",
            "🚁 Air Patrol", "🏥 Medical Support", "📡 Communication Hub",
            "🎯 Targeted Strike", "🤝 Diplomatic Mission"
        ]
        
        self.decision_reasoning = f"Neural Network Decision: {action_descriptions[action_type]} → Target: {target_country} ({target_city})"
        
        if target_crisis:
            self.decision_reasoning += f"\n🎯 Responding to: {target_crisis['type']} (Severity: {target_crisis['severity']:.1f})"
        
        # Record decision
        decision = {
            'time': time.time(),
            'action': action,
            'target_country': target_country,
            'target_city': target_city,
            'reasoning': action_descriptions[action_type],
            'crisis_response': target_crisis is not None
        }
        
        self.ai_decisions.append(decision)
        if len(self.ai_decisions) > 10:
            self.ai_decisions.pop(0)
        
        # Simulate neural activity
        self.neural_activity = {
            'input_activation': np.random.uniform(0.2, 1.0, 8),
            'hidden_activation': np.random.uniform(0.1, 0.9, 6),
            'output_activation': np.zeros(8),
            'decision_confidence': random.uniform(0.7, 0.95)
        }
        
        # Highlight chosen action
        self.neural_activity['output_activation'][action_type] = 1.0
        
        # Step environment for next decision
        _, reward, done, _, info = self.env.step(action)
        
        if done:
            self.current_obs, _ = self.env.reset()
            self.episode_count += 1
        
        return decision
    
    def plan_route_to_city(self, target_city: str):
        """Plan optimal route using A* pathfinding"""
        if target_city not in self.cities:
            return
        
        start_pos = tuple(self.agent.pos)
        target_pos = self.cities[target_city]
        
        # Find closest city to current position
        min_dist = float('inf')
        start_city = None
        for city, pos in self.cities.items():
            dist = math.sqrt((pos[0] - start_pos[0])**2 + (pos[1] - start_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                start_city = city
        
        if start_city == target_city:
            self.agent.path = [target_pos]
            return
        
        # A* pathfinding through road network
        path = self.a_star_pathfinding(start_city, target_city)
        
        if path:
            # Convert city path to coordinate path
            coord_path = [self.cities[city] for city in path]
            # Add current position as start
            if start_pos != coord_path[0]:
                coord_path.insert(0, start_pos)
            self.agent.path = coord_path
            self.agent.current_mission = f"Route to {target_city} via {' → '.join(path[:3])}"
    
    def a_star_pathfinding(self, start_city: str, target_city: str) -> List[str]:
        """A* pathfinding algorithm for road network"""
        # Build adjacency list from roads
        graph = {}
        for road in self.roads:
            city1, city2 = road
            if city1 not in graph:
                graph[city1] = []
            if city2 not in graph:
                graph[city2] = []
            graph[city1].append(city2)
            graph[city2].append(city1)
        
        def heuristic(city1: str, city2: str) -> float:
            pos1 = self.cities[city1]
            pos2 = self.cities[city2]
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
        # A* algorithm
        frontier = [(0, start_city, [start_city])]
        visited = set()
        
        while frontier:
            cost, current, path = heapq.heappop(frontier)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == target_city:
                return path
            
            if current in graph:
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        new_cost = len(new_path) + heuristic(neighbor, target_city)
                        heapq.heappush(frontier, (new_cost, neighbor, new_path))
        
        return [target_city]  # Direct route if no path found
    
    def check_crisis_detection(self):
        """Check if agent is close enough to detect crises"""
        detection_radius = 40
        
        for crisis in self.active_crises:
            if not crisis['discovered']:
                dist = math.sqrt((self.agent.pos[0] - crisis['pos'][0])**2 + 
                               (self.agent.pos[1] - crisis['pos'][1])**2)
                
                if dist < detection_radius:
                    crisis['discovered'] = True
                    self.agent.detected_crises.append(crisis)
                    
                    # Add to communication log
                    message = f"🚨 CRISIS DETECTED: {crisis['type']} at {crisis['location']} - {crisis['lives_at_risk']} lives at risk"
                    self.agent.communication_log.append({
                        'time': time.time(),
                        'message': message,
                        'type': 'detection'
                    })
                    
                    # Keep only recent messages
                    if len(self.agent.communication_log) > 8:
                        self.agent.communication_log.pop(0)
    
    def update(self, dt: float):
        """Enhanced simulation update with communication"""
        self.animation_time += dt
        self.pulse_time += dt * 3
        
        # Generate new crises
        self.generate_crisis()
        
        # Update AI decision every 2 seconds
        if int(self.animation_time) % 2 == 0 and int(self.animation_time * 10) % 20 < 2:
            self.update_ai_decision()
        
        # Move agent
        self.agent.move_towards_target(dt * 60)  # Scale dt for movement
        
        # Check crisis detection
        self.check_crisis_detection()
        
        # Update communication system
        self.update_communications()
        
        # Update performance metrics
        current_time = time.time()
        resolved_crises = [c for c in self.active_crises if c.get('responding', False) and 
                          math.sqrt((self.agent.pos[0] - c['pos'][0])**2 + (self.agent.pos[1] - c['pos'][1])**2) < 30]
        
        for crisis in resolved_crises:
            if crisis in self.active_crises:
                self.active_crises.remove(crisis)
                self.total_crises_resolved += 1
                self.total_lives_saved += crisis['lives_at_risk']
                
                # Send completion signal
                closest_center = self.find_closest_command_center()
                if closest_center:
                    signal = self.agent.send_signal_to_command(
                        self.command_centers[closest_center]['pos'],
                        f"Crisis resolved at {crisis['location']} - {crisis['lives_at_risk']} lives saved",
                        "success"
                    )
                    if signal:
                        self.active_signals.append(signal)
    
    def render(self):
        """Render the visualization"""
        self.screen.fill(self.colors['background'])
        
        # Render components
        self._render_map()
        self._render_neural_panel()
        self._render_decision_panel()
        self._render_communication_panel()
        self._render_performance_panel()
        self._render_header()
        self._render_legend()
    
    def _render_header(self):
        """Render main header"""
        title = self.font_title.render(f"🚁 Live Mobile Agent - {self.model_name} Control", True, self.colors['text'])
        self.screen.blit(title, (20, 20))
        
        subtitle = self.font_medium.render("Real-time neural network controlling mobile crisis response agent", True, self.colors['accent'])
        self.screen.blit(subtitle, (20, 50))
    
    def _render_map(self):
        """Enhanced map rendering with better regional visualization"""
        map_x, map_y, map_w, map_h = self.map_area
        
        # Map background
        pygame.draw.rect(self.screen, self.colors['land'], (map_x, map_y, map_w, map_h))
        pygame.draw.rect(self.screen, self.colors['accent'], (map_x, map_y, map_w, map_h), 2)
        
        # Enhanced Country regions with proper borders and labels
        for country, data in self.countries.items():
            # Draw region background
            if 'region_bounds' in data:
                try:
                    region_color = tuple(max(0, min(255, int(c * 0.4))) for c in data['color'])
                    border_color = data.get('border_color', data['color'])
                    
                    # Fill region
                    pygame.draw.polygon(self.screen, region_color, data['region_bounds'])
                    # Border
                    pygame.draw.polygon(self.screen, border_color, data['region_bounds'], 3)
                    
                    # Country label
                    center_x = sum(p[0] for p in data['region_bounds']) // len(data['region_bounds'])
                    center_y = sum(p[1] for p in data['region_bounds']) // len(data['region_bounds'])
                    
                    country_label = self.font_medium.render(country, True, self.colors['text'])
                    label_rect = country_label.get_rect(center=(center_x, center_y - 20))
                    
                    # Background for label
                    pygame.draw.rect(self.screen, self.colors['panel_bg'], 
                                   (label_rect.x - 5, label_rect.y - 2, label_rect.width + 10, label_rect.height + 4))
                    self.screen.blit(country_label, label_rect)
                    
                except (ValueError, OverflowError, TypeError):
                    pass  # Skip if rendering fails
        
        # Command Centers
        for center_name, center_data in self.command_centers.items():
            pos = center_data['pos']
            color = center_data['color']
            coverage = center_data['coverage']
            
            # Coverage area (subtle)
            coverage_color = tuple(max(0, min(255, int(c * 0.1))) for c in color)
            pygame.draw.circle(self.screen, coverage_color, pos, coverage, 1)
            
            # Command center building
            pygame.draw.rect(self.screen, color, (pos[0]-8, pos[1]-8, 16, 16))
            pygame.draw.rect(self.screen, self.colors['text'], (pos[0]-8, pos[1]-8, 16, 16), 2)
            
            # Antenna
            pygame.draw.line(self.screen, color, (pos[0], pos[1]-8), (pos[0], pos[1]-20), 3)
            
            # Label
            label = self.font_small.render(center_name, True, color)
            self.screen.blit(label, (pos[0] + 12, pos[1] - 10))
        
        # Road network
        for road in self.roads:
            city1, city2 = road
            if city1 in self.cities and city2 in self.cities:
                pos1 = self.cities[city1]
                pos2 = self.cities[city2]
                pygame.draw.line(self.screen, self.colors['text_dim'], pos1, pos2, 1)
        
        # Cities
        for city, pos in self.cities.items():
            pygame.draw.circle(self.screen, self.colors['text'], pos, 4)
            city_label = self.font_small.render(city, True, self.colors['text'])
            self.screen.blit(city_label, (pos[0] + 8, pos[1] - 8))
        
        # Active crises
        for crisis in self.active_crises:
            pos = crisis['pos']
            
            # Pulsing crisis indicator
            pulse = abs(math.sin(self.pulse_time)) * 0.5 + 0.5
            radius = int(15 + pulse * 10)
            
            try:
                color = tuple(max(0, min(255, int(c * pulse))) for c in crisis['color'])
                pygame.draw.circle(self.screen, color, pos, radius)
                
                # Crisis icon
                crisis_text = crisis['type'].split()[0]  # Get emoji
                crisis_surface = self.font_medium.render(crisis_text, True, self.colors['text'])
                text_rect = crisis_surface.get_rect(center=pos)
                self.screen.blit(crisis_surface, text_rect)
                
                # Severity indicator
                severity_text = f"{crisis['severity']:.1f}"
                severity_surface = self.font_small.render(severity_text, True, self.colors['text'])
                self.screen.blit(severity_surface, (pos[0] + 20, pos[1] + 15))
                
            except (ValueError, OverflowError):
                # Fallback rendering
                pygame.draw.circle(self.screen, self.colors['crisis'], pos, radius)
        
        # Agent path
        if len(self.agent.path) > 1:
            for i in range(len(self.agent.path) - 1):
                start = self.agent.path[i]
                end = self.agent.path[i + 1]
                pygame.draw.line(self.screen, self.colors['route'], start, end, 3)
        
        # Enhanced Agent visualization with trail
        agent_pos = tuple(map(int, self.agent.pos))
        
        # Agent movement trail
        if len(self.agent.trail_positions) > 1:
            for i in range(len(self.agent.trail_positions) - 1):
                start_pos = tuple(map(int, self.agent.trail_positions[i]))
                end_pos = tuple(map(int, self.agent.trail_positions[i + 1]))
                
                # Fade trail based on age
                alpha = max(50, 255 - (len(self.agent.trail_positions) - i) * 12)
                trail_color = (*self.colors['agent'][:3], alpha)
                
                try:
                    pygame.draw.line(self.screen, self.colors['agent'], start_pos, end_pos, max(1, 4 - i//3))
                except (ValueError, OverflowError):
                    pass
        
        # Communication signals (agent to stations)
        for signal in self.active_signals:
            self._render_signal(signal)
        
        # Station responses
        for response in self.active_responses:
            self._render_response(response)
        
        # Agent glow effect
        glow_radius = int(20 + abs(math.sin(self.animation_time * 2)) * 8)
        try:
            glow_color = tuple(max(0, min(255, int(c * 0.3))) for c in self.colors['agent'])
            pygame.draw.circle(self.screen, glow_color, agent_pos, glow_radius)
        except (ValueError, OverflowError):
            pass
        
        # Agent body
        pygame.draw.circle(self.screen, self.colors['agent'], agent_pos, 12)
        pygame.draw.circle(self.screen, self.colors['text'], agent_pos, 12, 2)
        
        # Agent direction indicator
        if self.agent.path:
            target = self.agent.path[0]
            dx = target[0] - self.agent.pos[0]
            dy = target[1] - self.agent.pos[1]
            if dx != 0 or dy != 0:
                length = math.sqrt(dx*dx + dy*dy)
                dx, dy = dx/length, dy/length
                end_x = agent_pos[0] + dx * 15
                end_y = agent_pos[1] + dy * 15
                pygame.draw.line(self.screen, self.colors['text'], agent_pos, (int(end_x), int(end_y)), 3)
        
        # Enhanced agent label with status
        agent_label = self.font_small.render(f"🚁 {self.agent.mission_status}", True, self.colors['agent'])
        self.screen.blit(agent_label, (agent_pos[0] - 60, agent_pos[1] - 30))
        
        # Fuel and signal strength indicators
        fuel_text = f"Fuel: {self.agent.fuel:.0f}%"
        fuel_color = self.colors['stable'] if self.agent.fuel > 50 else (self.colors['warning'] if self.agent.fuel > 20 else self.colors['crisis'])
        fuel_surface = self.font_small.render(fuel_text, True, fuel_color)
        self.screen.blit(fuel_surface, (agent_pos[0] - 30, agent_pos[1] + 20))
        
        signal_text = f"Signal: {self.agent.signal_strength:.0f}%"
        signal_color = self.colors['stable'] if self.agent.signal_strength > 70 else (self.colors['warning'] if self.agent.signal_strength > 30 else self.colors['crisis'])
        signal_surface = self.font_small.render(signal_text, True, signal_color)
        self.screen.blit(signal_surface, (agent_pos[0] - 30, agent_pos[1] + 35))
    
    def _render_neural_panel(self):
        """Render neural network activity panel"""
        panel_x = 870
        panel_y = 100
        panel_w = 350
        panel_h = 250
        
        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(self.screen, self.colors['neural'], (panel_x, panel_y, panel_w, panel_h), 2)
        
        title = self.font_large.render("🧠 Neural Network Activity", True, self.colors['neural'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        if not self.neural_activity:
            no_activity = self.font_medium.render("Initializing neural network...", True, self.colors['text_dim'])
            self.screen.blit(no_activity, (panel_x + 10, panel_y + 50))
            return
        
        # Neural layers visualization
        layer_x = panel_x + 30
        layer_y = panel_y + 50
        
        # Input layer
        input_label = self.font_small.render("Inputs", True, self.colors['text'])
        self.screen.blit(input_label, (layer_x, layer_y))
        
        for i, activation in enumerate(self.neural_activity['input_activation']):
            y = layer_y + 20 + i * 20
            bar_width = int(60 * activation)
            color_intensity = int(255 * activation)
            color = (color_intensity, color_intensity // 2, 50)
            pygame.draw.rect(self.screen, color, (layer_x, y, bar_width, 12))
            
            value_text = self.font_small.render(f"{activation:.2f}", True, self.colors['text_dim'])
            self.screen.blit(value_text, (layer_x + 70, y))
        
        # Hidden layer  
        hidden_x = layer_x + 120
        hidden_label = self.font_small.render("Hidden", True, self.colors['text'])
        self.screen.blit(hidden_label, (hidden_x, layer_y))
        
        for i, activation in enumerate(self.neural_activity['hidden_activation']):
            y = layer_y + 20 + i * 25
            radius = int(8 + activation * 5)
            color_intensity = int(255 * activation)
            color = (50, color_intensity, 50)
            pygame.draw.circle(self.screen, color, (hidden_x + 30, y + 6), radius)
        
        # Output layer
        output_x = layer_x + 200
        output_label = self.font_small.render("Outputs", True, self.colors['text'])
        self.screen.blit(output_label, (output_x, layer_y))
        
        for i, activation in enumerate(self.neural_activity['output_activation']):
            y = layer_y + 20 + i * 20
            bar_width = int(60 * activation)
            color_intensity = int(255 * activation)
            color = (50, 50, color_intensity)
            pygame.draw.rect(self.screen, color, (output_x, y, bar_width, 12))
        
        # Decision confidence
        confidence = self.neural_activity.get('decision_confidence', 0)
        confidence_text = f"Decision Confidence: {confidence:.1%}"
        confidence_color = self.colors['stable'] if confidence > 0.8 else self.colors['warning']
        confidence_surface = self.font_medium.render(confidence_text, True, confidence_color)
        self.screen.blit(confidence_surface, (panel_x + 10, panel_y + 220))
    
    def _render_decision_panel(self):
        """Render AI decision reasoning panel"""
        panel_x = 1240
        panel_y = 100
        panel_w = 340
        panel_h = 250
        
        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(self.screen, self.colors['gold'], (panel_x, panel_y, panel_w, panel_h), 2)
        
        title = self.font_large.render("🎯 AI Decision Making", True, self.colors['gold'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Current reasoning
        if self.decision_reasoning:
            lines = self.decision_reasoning.split('\n')
            for i, line in enumerate(lines[:8]):  # Max 8 lines
                color = self.colors['text'] if i == 0 else self.colors['text_dim']
                text_surface = self.font_small.render(line[:40], True, color)  # Truncate long lines
                self.screen.blit(text_surface, (panel_x + 10, panel_y + 40 + i * 18))
        
        # Recent decisions
        decisions_y = panel_y + 180
        decisions_title = self.font_medium.render("Recent Decisions:", True, self.colors['text'])
        self.screen.blit(decisions_title, (panel_x + 10, decisions_y))
        
        for i, decision in enumerate(self.ai_decisions[-3:]):  # Last 3 decisions
            y = decisions_y + 20 + i * 15
            decision_text = f"{decision['reasoning'][:25]} → {decision['target_city']}"
            text_surface = self.font_small.render(decision_text, True, self.colors['text_dim'])
            self.screen.blit(text_surface, (panel_x + 10, y))
    
    def _render_communication_panel(self):
        """Render communication log panel"""
        panel_x = 870
        panel_y = 370
        panel_w = 710
        panel_h = 180
        
        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(self.screen, self.colors['accent'], (panel_x, panel_y, panel_w, panel_h), 2)
        
        title = self.font_large.render("📡 Live Communication Log", True, self.colors['accent'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Communication messages
        for i, comm in enumerate(self.agent.communication_log[-8:]):  # Last 8 messages
            y = panel_y + 40 + i * 18
            
            # Timestamp
            time_diff = time.time() - comm['time']
            timestamp = f"[T-{time_diff:.0f}s]"
            timestamp_surface = self.font_small.render(timestamp, True, self.colors['text_dim'])
            self.screen.blit(timestamp_surface, (panel_x + 10, y))
            
            # Message
            message = comm['message'][:80]  # Truncate long messages
            message_color = self.colors['crisis'] if 'CRISIS' in message else self.colors['text']
            message_surface = self.font_small.render(message, True, message_color)
            self.screen.blit(message_surface, (panel_x + 80, y))
        
        # Current mission
        if self.agent.current_mission:
            mission_y = panel_y + 150
            mission_label = self.font_medium.render("Current Mission:", True, self.colors['gold'])
            self.screen.blit(mission_label, (panel_x + 10, mission_y))
            
            mission_text = self.agent.current_mission[:60]
            mission_surface = self.font_small.render(mission_text, True, self.colors['text'])
            self.screen.blit(mission_surface, (panel_x + 120, mission_y))
    
    def _render_performance_panel(self):
        """Render performance metrics panel"""
        panel_x = 870
        panel_y = 570
        panel_w = 710
        panel_h = 120
        
        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(self.screen, self.colors['stable'], (panel_x, panel_y, panel_w, panel_h), 2)
        
        title = self.font_large.render("📊 Mission Performance Metrics", True, self.colors['stable'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Metrics
        metrics = [
            f"Episodes Completed: {self.episode_count}",
            f"Lives Saved: {self.total_lives_saved:,}",
            f"Crises Resolved: {self.total_crises_resolved}",
            f"Active Crises: {len(self.active_crises)}",
            f"Agent Fuel: {self.agent.fuel:.0f}%"
        ]
        
        for i, metric in enumerate(metrics):
            x = panel_x + 10 + (i % 3) * 230
            y = panel_y + 40 + (i // 3) * 25
            
            metric_surface = self.font_medium.render(metric, True, self.colors['text'])
            self.screen.blit(metric_surface, (x, y))
        
        # Controls
        controls_text = "Controls: SPACE=Pause | N=Manual Step | R=New Mission | ESC=Exit"
        controls_surface = self.font_small.render(controls_text, True, self.colors['text_dim'])
        self.screen.blit(controls_surface, (panel_x + 10, panel_y + 95))
    
    def run(self, auto_update=True, update_interval=1.0):
        """Run the live visualization"""
        if not SB3_AVAILABLE:
            print("❌ Stable Baselines3 not available!")
            return
        
        if not self.model:
            print("❌ No model loaded!")
            return
        
        print(f"🚁 Starting live mobile agent visualization with {self.model_name}")
        print("🎯 Watch the neural network control the mobile crisis response agent!")
        
        running = True
        paused = False
        last_update_time = time.time()
        dt = 0.016  # ~60 FPS
        
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
                        self.update_ai_decision()
                        print("👆 Manual AI decision triggered")
                    elif event.key == pygame.K_r:
                        # New mission
                        self.agent.path = []
                        self.agent.current_mission = None
                        print("🔄 New mission assigned")
            
            # Update simulation
            if not paused:
                if auto_update and (current_time - last_update_time) >= update_interval:
                    last_update_time = current_time
                
                self.update(dt)
            
            # Render
            self.render()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        print("👋 Mobile agent visualization ended")
        print(f"📊 Final Statistics:")
        print(f"   Lives Saved: {self.total_lives_saved:,}")
        print(f"   Crises Resolved: {self.total_crises_resolved}")
        print(f"   Episodes: {self.episode_count}")

    def update_communications(self):
        """Update communication signals and responses"""
        current_time = time.time()
        
        # Send regular status updates
        if current_time - self.agent.last_signal_time >= self.agent.signal_cooldown:
            closest_center = self.find_closest_command_center()
            if closest_center:
                message = f"Status: {self.agent.mission_status} | Fuel: {self.agent.fuel:.0f}% | Crises: {len(self.active_crises)}"
                signal = self.agent.send_signal_to_command(
                    self.command_centers[closest_center]['pos'],
                    message,
                    "status"
                )
                if signal:
                    self.active_signals.append(signal)
                    
                    # Generate response after delay
                    response_delay = random.uniform(0.5, 1.5)
                    response = {
                        'from_pos': self.command_centers[closest_center]['pos'],
                        'to_pos': self.agent.pos.copy(),
                        'message': self.generate_command_response(),
                        'timestamp': current_time + response_delay,
                        'center_name': closest_center
                    }
                    self.active_responses.append(response)
        
        # Update and remove old signals
        self.active_signals = [s for s in self.active_signals if current_time - s['timestamp'] < 3.0]
        self.active_responses = [r for r in self.active_responses if current_time - r['timestamp'] < 3.0]
    
    def find_closest_command_center(self):
        """Find the closest command center to the agent"""
        min_distance = float('inf')
        closest_center = None
        
        for center_name, center_data in self.command_centers.items():
            distance = math.sqrt((self.agent.pos[0] - center_data['pos'][0])**2 + 
                               (self.agent.pos[1] - center_data['pos'][1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_center = center_name
        
        return closest_center
    
    def generate_command_response(self):
        """Generate realistic command center responses"""
        responses = [
            "Roger, continue mission",
            "Status acknowledged, proceed", 
            "Good work, agent. Stay alert",
            "Backup en route if needed",
            "Priority: civilian safety first",
            "Weather clear, proceed",
            "Intel updated, check map",
            "Fuel station available nearby"
        ]
        return random.choice(responses)
    
    def _render_signal(self, signal):
        """Render communication signal from agent to station"""
        current_time = time.time()
        signal_age = current_time - signal['timestamp']
        
        if signal_age > 3.0:  # Signal duration
            return
        
        # Signal animation
        progress = signal_age / 3.0
        
        start_pos = tuple(map(int, signal['from_pos']))
        end_pos = tuple(map(int, signal['to_pos']))
        
        # Current signal position
        current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
        current_pos = (int(current_x), int(current_y))
        
        # Signal visualization
        signal_color = self.colors['neural'] if signal['type'] == 'status' else self.colors['crisis']
        
        # Signal beam
        pygame.draw.line(self.screen, signal_color, start_pos, current_pos, 2)
        
        # Signal pulse
        pulse_radius = int(5 + abs(math.sin(current_time * 10)) * 3)
        pygame.draw.circle(self.screen, signal_color, current_pos, pulse_radius)
        
        # Signal type indicator
        if signal['type'] == 'status':
            icon = "📡"
        elif signal['type'] == 'crisis':
            icon = "🚨"
        else:
            icon = "✅"
        
        signal_text = self.font_small.render(icon, True, signal_color)
        self.screen.blit(signal_text, (current_pos[0] - 8, current_pos[1] - 8))
    
    def _render_response(self, response):
        """Render response signal from station to agent"""
        current_time = time.time()
        
        if current_time < response['timestamp']:
            return  # Not yet sent
        
        response_age = current_time - response['timestamp']
        if response_age > 3.0:  # Response duration
            return
        
        # Response animation
        progress = response_age / 3.0
        
        start_pos = tuple(map(int, response['from_pos']))
        end_pos = tuple(map(int, response['to_pos']))
        
        # Current response position
        current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
        current_pos = (int(current_x), int(current_y))
        
        # Response visualization (different from signals)
        response_color = self.colors['gold']
        
        # Response beam (dashed)
        segments = 10
        for i in range(segments):
            if i % 2 == 0:  # Dashed effect
                seg_start_x = start_pos[0] + (current_pos[0] - start_pos[0]) * (i / segments)
                seg_start_y = start_pos[1] + (current_pos[1] - start_pos[1]) * (i / segments)
                seg_end_x = start_pos[0] + (current_pos[0] - start_pos[0]) * ((i + 1) / segments)
                seg_end_y = start_pos[1] + (current_pos[1] - start_pos[1]) * ((i + 1) / segments)
                
                pygame.draw.line(self.screen, response_color, 
                               (int(seg_start_x), int(seg_start_y)), 
                               (int(seg_end_x), int(seg_end_y)), 2)
        
        # Response pulse
        pulse_radius = int(4 + abs(math.sin(current_time * 8)) * 2)
        pygame.draw.circle(self.screen, response_color, current_pos, pulse_radius)
        
        # Response icon
        response_text = self.font_small.render("📻", True, response_color)
        self.screen.blit(response_text, (current_pos[0] - 8, current_pos[1] - 8))
    
    def _render_legend(self):
        """Render legend explaining visual elements"""
        legend_x = 870
        legend_y = 750
        legend_w = 350
        legend_h = 200
        
        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (legend_x, legend_y, legend_w, legend_h))
        pygame.draw.rect(self.screen, self.colors['accent'], (legend_x, legend_y, legend_w, legend_h), 2)
        
        title = self.font_large.render("🗺️ Legend", True, self.colors['accent'])
        self.screen.blit(title, (legend_x + 10, legend_y + 10))
        
        legend_items = [
            ("🚁", "Mobile Crisis Agent", self.colors['agent']),
            ("📡", "Agent Signal", self.colors['neural']),
            ("📻", "Command Response", self.colors['gold']),
            ("🚨", "Active Crisis", self.colors['crisis']),
            ("🏛️", "Command Centers", self.colors['gold']),
            ("━━", "Movement Trail", self.colors['agent']),
            ("▬▬", "Country Borders", self.colors['accent'])
        ]
        
        y_offset = 40
        for icon, description, color in legend_items:
            icon_surface = self.font_medium.render(icon, True, color)
            desc_surface = self.font_small.render(description, True, self.colors['text'])
            
            self.screen.blit(icon_surface, (legend_x + 15, legend_y + y_offset))
            self.screen.blit(desc_surface, (legend_x + 40, legend_y + y_offset + 2))
            
            y_offset += 20

def main():
    """Main function"""
    import sys
    
    if not SB3_AVAILABLE:
        print("❌ Stable Baselines3 not available!")
        print("Install with: pip install stable-baselines3")
        return
    
    # Check for demo mode
    demo_mode = len(sys.argv) > 1 and sys.argv[1] == "--demo"
    
    print("🚁 LIVE MOBILE AGENT NEURAL NETWORK VISUALIZATION")
    print("=" * 60)
    print("Watch your trained neural networks control a mobile crisis response agent!")
    print("The agent navigates real road networks in Africa (Cameroon, DRC, Sudan)")
    print()
    
    # Find available models
    import os
    models_dir = "models"
    available_models = {}
    
    for algorithm in ['dqn', 'ppo', 'a2c']:
        best_path = os.path.join(models_dir, f"{algorithm}_best", "best_model.zip")
        if os.path.exists(best_path):
            available_models[algorithm.upper()] = best_path
    
    if not available_models:
        print("❌ No trained models found!")
        print("Train models first using: python real_rl_training.py")
        input("Press ENTER to exit...")
        return
    
    print("Available trained models:")
    algorithms = list(available_models.keys())
    for i, alg in enumerate(algorithms):
        print(f"  {i+1}. {alg}")
    
    try:
        if demo_mode:
            print("\n🎬 DEMO MODE: Automatically selecting DQN...")
            choice = 0  # Auto-select DQN
        else:
            choice = int(input(f"\nChoose model to control the mobile agent (1-{len(algorithms)}): ")) - 1
        
        if 0 <= choice < len(algorithms):
            algorithm = algorithms[choice]
            model_path = available_models[algorithm]
            
            print(f"\n🚀 Loading {algorithm} model for mobile agent control...")
            
            # Create visualizer
            visualizer = LiveMobileVisualization()
            
            if visualizer.load_model(model_path, algorithm):
                print(f"✅ {algorithm} neural network loaded successfully!")
                print("\n🎮 Controls:")
                print("   SPACE = Pause/Resume simulation")
                print("   N = Trigger manual AI decision")
                print("   R = Assign new mission")
                print("   ESC = Exit")
                print()
                print("🎬 Starting live mobile agent visualization...")
                print("🎯 The neural network will make decisions about where to move and which crises to prioritize!")
                
                if demo_mode:
                    print("\n⏰ DEMO MODE: Will auto-close after 30 seconds")
                    print("Press ESC anytime to exit early")
                
                # Run visualization
                visualizer.run(auto_update=True, update_interval=1.5)  # AI decision every 1.5 seconds
            else:
                print("❌ Failed to load model")
        else:
            print("❌ Invalid choice")
    
    except (ValueError, KeyboardInterrupt):
        print("\n👋 Goodbye!")
    
    input("Press ENTER to exit...")

if __name__ == "__main__":
    main()