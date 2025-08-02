#!/usr/bin/env python3
"""
Mobile Crisis Response Agent - Dynamic Navigation System
Cameroon, DRC, Sudan Crisis Response with Moving Agent

A moving AI agent that physically navigates through road networks,
discovers crises, and coordinates optimal response routes.
"""

import pygame
import numpy as np
import math
import time
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import heapq

class CrisisType(Enum):
    ARMED_CONFLICT = "Armed Conflict"
    REFUGEE_CAMP = "Refugee Emergency"
    HUMANITARIAN = "Humanitarian Crisis"
    POLITICAL_UNREST = "Political Unrest"
    RESOURCE_CONFLICT = "Resource Conflict"
    BORDER_TENSION = "Border Tension"

class AgentStatus(Enum):
    PATROLLING = "Patrolling"
    INVESTIGATING = "Investigating Crisis"
    TRAVELING_TO_CRISIS = "En Route to Crisis"
    REQUESTING_SUPPORT = "Requesting Support"
    COORDINATING_RESPONSE = "Coordinating Response"

@dataclass
class Location:
    """Represents a location on the map"""
    name: str
    x: float
    y: float
    is_city: bool = False
    population: int = 0
    risk_level: float = 0.0

@dataclass
class Road:
    """Represents a road connection between locations"""
    start: str
    end: str
    distance: float
    condition: float  # 0-1, 1 = excellent roads
    danger_level: float  # 0-1, 1 = very dangerous

@dataclass
class Crisis:
    """Represents a discovered crisis"""
    id: int
    location: str
    crisis_type: CrisisType
    severity: float  # 0-1
    urgency: float  # 0-1
    affected_population: int
    discovered_time: float
    coordinates: Tuple[float, float]
    response_sent: bool = False

@dataclass
class ResponseUnit:
    """Represents a response unit sent to help"""
    unit_type: str
    route: List[str]
    travel_time: float
    status: str

class PathFinder:
    """A* pathfinding for optimal route calculation"""
    
    def __init__(self, roads: List[Road], locations: Dict[str, Location]):
        self.roads = roads
        self.locations = locations
        self.graph = self._build_graph()
    
    def _build_graph(self) -> Dict[str, List[Tuple[str, float]]]:
        """Build adjacency graph from roads"""
        graph = {}
        for road in self.roads:
            if road.start not in graph:
                graph[road.start] = []
            if road.end not in graph:
                graph[road.end] = []
            
            # Weight includes distance and road conditions
            weight = road.distance * (2.0 - road.condition) * (1.0 + road.danger_level)
            
            graph[road.start].append((road.end, weight))
            graph[road.end].append((road.start, weight))  # Bidirectional
        
        return graph
    
    def find_path(self, start: str, end: str) -> Tuple[List[str], float]:
        """Find optimal path using A* algorithm"""
        if start not in self.graph or end not in self.graph:
            return [], float('inf')
        
        if start == end:
            return [start], 0.0
        
        # A* implementation
        open_set = [(0, start, [start], 0)]
        closed_set = set()
        
        while open_set:
            _, current, path, cost = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            if current == end:
                return path, cost
            
            for neighbor, edge_cost in self.graph.get(current, []):
                if neighbor in closed_set:
                    continue
                
                new_cost = cost + edge_cost
                heuristic = self._heuristic(neighbor, end)
                priority = new_cost + heuristic
                
                new_path = path + [neighbor]
                heapq.heappush(open_set, (priority, neighbor, new_path, new_cost))
        
        return [], float('inf')  # No path found
    
    def _heuristic(self, loc1: str, loc2: str) -> float:
        """Euclidean distance heuristic"""
        if loc1 not in self.locations or loc2 not in self.locations:
            return 0
        
        p1 = self.locations[loc1]
        p2 = self.locations[loc2]
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2) * 0.1

class MobileCrisisAgent:
    """Mobile AI agent that moves through the map and discovers crises"""
    
    def __init__(self, start_location: str, locations: Dict[str, Location], pathfinder: PathFinder):
        self.current_location = start_location
        self.locations = locations
        self.pathfinder = pathfinder
        
        # Agent state
        self.status = AgentStatus.PATROLLING
        self.target_location: Optional[str] = None
        self.current_path: List[str] = []
        self.path_progress = 0.0
        self.movement_speed = 2.0  # pixels per frame
        
        # Agent capabilities
        self.detection_radius = 80.0  # pixels
        self.communication_range = 200.0  # pixels to HQ
        
        # Mission data
        self.crises_discovered: List[Crisis] = []
        self.response_units: List[ResponseUnit] = []
        self.total_distance_traveled = 0.0
        self.mission_time = 0.0
        
        # Current position (for smooth movement)
        start_pos = self.locations[start_location]
        self.x = start_pos.x
        self.y = start_pos.y

class AfricaMobileSystem:
    """Complete mobile crisis response system"""
    
    def __init__(self, width=1400, height=900):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🚁 Mobile Crisis Response Agent - Africa Operations")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.colors = {
            'background': (15, 25, 35),
            'land': (40, 60, 40),
            'road': (80, 80, 80),
            'highway': (120, 120, 120),
            'city': (255, 255, 100),
            'agent': (50, 255, 150),
            'crisis_low': (255, 200, 50),
            'crisis_high': (255, 50, 50),
            'response_unit': (100, 150, 255),
            'communication': (0, 255, 255),
            'hq': (255, 100, 255),
            'text': (255, 255, 255),
            'panel_bg': (25, 35, 45)
        }
        
        # Initialize map data
        self.locations = self._create_locations()
        self.roads = self._create_road_network()
        self.pathfinder = PathFinder(self.roads, self.locations)
        
        # Initialize agent
        self.agent = MobileCrisisAgent("Douala", self.locations, self.pathfinder)
        
        # Crisis system
        self.active_crises: List[Crisis] = []
        self.crisis_id_counter = 0
        
        # Fonts
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 16)
        
        # Animation
        self.animation_time = 0
        self.message_queue: List[Dict] = []
        
        print("🚁 Mobile Crisis Response System Initialized")
        print("🌍 Coverage: Cameroon, DR Congo, Sudan")
        print("🎯 Agent starting patrol from Douala, Cameroon")
    
    def _create_locations(self) -> Dict[str, Location]:
        """Create realistic locations for the three countries"""
        locations = {}
        
        # Cameroon locations
        cameroon_cities = [
            ("Douala", 200, 300, True, 3500000, 0.3),
            ("Yaoundé", 250, 280, True, 4000000, 0.4),
            ("Bamenda", 220, 200, True, 500000, 0.7),  # Anglophone crisis
            ("Maroua", 280, 150, True, 400000, 0.8),   # Boko Haram area
            ("Garoua", 260, 180, False, 300000, 0.6),
            ("Bafoussam", 230, 250, False, 400000, 0.4),
        ]
        
        # DR Congo locations  
        drc_cities = [
            ("Kinshasa", 500, 400, True, 15000000, 0.5),
            ("Lubumbashi", 600, 500, True, 2500000, 0.6),
            ("Mbuji-Mayi", 580, 450, True, 2000000, 0.7),
            ("Goma", 650, 350, True, 1000000, 0.9),     # M23 conflict zone
            ("Bukavu", 660, 380, True, 800000, 0.8),    # Armed groups
            ("Kisangani", 580, 300, True, 1500000, 0.7),
            ("Uvira", 670, 400, False, 200000, 0.9),    # Border tensions
        ]
        
        # Sudan locations
        sudan_cities = [
            ("Khartoum", 900, 200, True, 6000000, 0.95), # Capital conflict
            ("Omdurman", 880, 190, True, 3000000, 0.9),
            ("Port Sudan", 1000, 180, True, 700000, 0.6),
            ("Kassala", 1020, 220, True, 500000, 0.8),
            ("El Obeid", 850, 250, True, 400000, 0.7),
            ("Nyala", 820, 300, True, 500000, 0.8),     # Darfur
            ("El Fasher", 800, 280, False, 300000, 0.9), # Darfur conflict
        ]
        
        all_cities = cameroon_cities + drc_cities + sudan_cities
        
        for name, x, y, is_city, pop, risk in all_cities:
            locations[name] = Location(name, x, y, is_city, pop, risk)
        
        return locations
    
    def _create_road_network(self) -> List[Road]:
        """Create realistic road connections"""
        roads = []
        
        # Cameroon roads
        cameroon_roads = [
            ("Douala", "Yaoundé", 250, 0.8, 0.2),
            ("Yaoundé", "Bamenda", 300, 0.6, 0.4),
            ("Bamenda", "Maroua", 400, 0.4, 0.7),  # Dangerous northern route
            ("Yaoundé", "Bafoussam", 200, 0.7, 0.3),
            ("Bafoussam", "Bamenda", 150, 0.6, 0.4),
            ("Maroua", "Garoua", 180, 0.5, 0.6),
        ]
        
        # DR Congo roads
        drc_roads = [
            ("Kinshasa", "Lubumbashi", 800, 0.3, 0.5),
            ("Kinshasa", "Mbuji-Mayi", 600, 0.4, 0.6),
            ("Lubumbashi", "Mbuji-Mayi", 400, 0.5, 0.5),
            ("Goma", "Bukavu", 200, 0.3, 0.9),     # Very dangerous
            ("Bukavu", "Uvira", 150, 0.2, 0.9),
            ("Kisangani", "Goma", 500, 0.2, 0.8),
            ("Mbuji-Mayi", "Kisangani", 400, 0.3, 0.7),
        ]
        
        # Sudan roads
        sudan_roads = [
            ("Khartoum", "Omdurman", 50, 0.7, 0.9),   # War zone
            ("Khartoum", "Port Sudan", 800, 0.6, 0.5),
            ("Khartoum", "Kassala", 600, 0.5, 0.6),
            ("Port Sudan", "Kassala", 400, 0.6, 0.4),
            ("Khartoum", "El Obeid", 400, 0.4, 0.7),
            ("El Obeid", "Nyala", 500, 0.3, 0.8),
            ("Nyala", "El Fasher", 200, 0.2, 0.9),    # Darfur conflict
        ]
        
        # Cross-border connections (limited)
        border_roads = [
            ("Maroua", "El Fasher", 1200, 0.2, 0.8),   # Chad border route
            ("Uvira", "Kassala", 1000, 0.1, 0.9),      # Dangerous cross-border
        ]
        
        all_roads = cameroon_roads + drc_roads + sudan_roads + border_roads
        
        for start, end, dist, condition, danger in all_roads:
            roads.append(Road(start, end, dist, condition, danger))
        
        return roads
    
    def update_simulation(self):
        """Update the entire simulation"""
        self.animation_time += 0.05
        self.agent.mission_time += 1/30  # 30 FPS
        
        # Update agent
        self._update_agent()
        
        # Generate random crises
        if random.random() < 0.01:  # 1% chance per frame
            self._generate_crisis()
        
        # Update crises
        self._update_crises()
        
        # Update message queue
        self._update_messages()
    
    def _update_agent(self):
        """Update agent movement and decision making"""
        # Agent AI decision making
        if self.agent.status == AgentStatus.PATROLLING:
            self._agent_patrol_decision()
        elif self.agent.status == AgentStatus.TRAVELING_TO_CRISIS:
            self._agent_travel_to_crisis()
        elif self.agent.status == AgentStatus.INVESTIGATING:
            self._agent_investigate_crisis()
        
        # Physical movement
        self._update_agent_movement()
        
        # Crisis detection
        self._detect_nearby_crises()
    
    def _agent_patrol_decision(self):
        """Agent decides where to patrol next"""
        # Check for undiscovered crises
        undiscovered_crises = [c for c in self.active_crises if not any(
            dc.id == c.id for dc in self.agent.crises_discovered)]
        
        if undiscovered_crises:
            # Find the most urgent crisis
            most_urgent = max(undiscovered_crises, key=lambda c: c.urgency * c.severity)
            
            # Plan route to crisis
            path, distance = self.pathfinder.find_path(
                self.agent.current_location, most_urgent.location)
            
            if path:
                self.agent.current_path = path[1:]  # Exclude current location
                self.agent.path_progress = 0.0
                self.agent.target_location = most_urgent.location
                self.agent.status = AgentStatus.TRAVELING_TO_CRISIS
                
                self._add_message(f"🎯 Agent routing to {most_urgent.location}", 
                                "info", f"Estimated distance: {distance:.0f}km")
        else:
            # Random patrol to high-risk areas
            high_risk_locations = [name for name, loc in self.locations.items() 
                                 if loc.risk_level > 0.6 and name != self.agent.current_location]
            
            if high_risk_locations and random.random() < 0.02:  # 2% chance to move
                target = random.choice(high_risk_locations)
                path, distance = self.pathfinder.find_path(self.agent.current_location, target)
                
                if path:
                    self.agent.current_path = path[1:]
                    self.agent.path_progress = 0.0
                    self.agent.target_location = target
                    self.agent.status = AgentStatus.PATROLLING
    
    def _agent_travel_to_crisis(self):
        """Agent traveling to crisis location"""
        if not self.agent.current_path:
            # Arrived at crisis location
            self.agent.status = AgentStatus.INVESTIGATING
            self._add_message(f"🔍 Agent arrived at {self.agent.current_location}", 
                            "success", "Beginning crisis investigation")
    
    def _agent_investigate_crisis(self):
        """Agent investigating crisis at current location"""
        # Find crisis at current location
        local_crises = [c for c in self.active_crises 
                       if c.location == self.agent.current_location]
        
        if local_crises:
            for crisis in local_crises:
                if not any(dc.id == crisis.id for dc in self.agent.crises_discovered):
                    # Discover crisis
                    self.agent.crises_discovered.append(crisis)
                    
                    # Request support
                    self._request_support(crisis)
                    
                    self._add_message(f"🚨 CRISIS DISCOVERED: {crisis.crisis_type.value}", 
                                    "alert", f"Severity: {crisis.severity:.0%}, Population: {crisis.affected_population:,}")
        
        # Return to patrol after investigation
        if random.random() < 0.1:  # 10% chance to finish investigation
            self.agent.status = AgentStatus.PATROLLING
            self._add_message(f"📡 Investigation complete at {self.agent.current_location}", 
                            "info", "Resuming patrol operations")
    
    def _update_agent_movement(self):
        """Update agent physical movement"""
        if not self.agent.current_path:
            return
        
        # Get current and next location
        current_idx = min(len(self.agent.current_path) - 1, int(self.agent.path_progress))
        
        if current_idx >= len(self.agent.current_path):
            # Path completed
            self.agent.current_location = self.agent.current_path[-1]
            loc = self.locations[self.agent.current_location]
            self.agent.x = loc.x
            self.agent.y = loc.y
            self.agent.current_path = []
            self.agent.path_progress = 0.0
            return
        
        next_location_name = self.agent.current_path[current_idx]
        next_location = self.locations[next_location_name]
        
        # Move towards next location
        dx = next_location.x - self.agent.x
        dy = next_location.y - self.agent.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < self.agent.movement_speed:
            # Reached next waypoint
            self.agent.x = next_location.x
            self.agent.y = next_location.y
            self.agent.current_location = next_location_name
            self.agent.path_progress += 1.0
            self.agent.total_distance_traveled += distance
        else:
            # Move towards next waypoint
            self.agent.x += (dx / distance) * self.agent.movement_speed
            self.agent.y += (dy / distance) * self.agent.movement_speed
            self.agent.total_distance_traveled += self.agent.movement_speed
    
    def _detect_nearby_crises(self):
        """Detect crises within agent's detection radius"""
        for crisis in self.active_crises:
            # Check if crisis is near agent
            dx = crisis.coordinates[0] - self.agent.x
            dy = crisis.coordinates[1] - self.agent.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance <= self.agent.detection_radius:
                # Crisis detected
                if not any(dc.id == crisis.id for dc in self.agent.crises_discovered):
                    # Visual detection effect
                    self._add_message(f"👁️ Crisis detected nearby: {crisis.crisis_type.value}", 
                                    "warning", f"Location: {crisis.location}")
    
    def _generate_crisis(self):
        """Generate a new random crisis"""
        # Choose random location weighted by risk level
        location_names = list(self.locations.keys())
        weights = [self.locations[name].risk_level for name in location_names]
        
        if sum(weights) == 0:
            return
        
        # Weighted random selection
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        selected_location = location_names[0]
        
        for name, weight in zip(location_names, weights):
            cumulative += weight
            if r <= cumulative:
                selected_location = name
                break
        
        location = self.locations[selected_location]
        
        # Create crisis
        crisis_types = list(CrisisType)
        crisis_type = random.choice(crisis_types)
        
        crisis = Crisis(
            id=self.crisis_id_counter,
            location=selected_location,
            crisis_type=crisis_type,
            severity=random.uniform(0.3, 1.0),
            urgency=random.uniform(0.4, 1.0),
            affected_population=random.randint(1000, int(location.population * 0.1)),
            discovered_time=self.agent.mission_time,
            coordinates=(location.x + random.randint(-20, 20), 
                        location.y + random.randint(-20, 20))
        )
        
        self.active_crises.append(crisis)
        self.crisis_id_counter += 1
        
        self._add_message(f"⚠️ New crisis emerging in {selected_location}", 
                        "warning", f"{crisis_type.value}")
    
    def _request_support(self, crisis: Crisis):
        """Request support for discovered crisis"""
        # Determine response type based on crisis
        if crisis.crisis_type in [CrisisType.ARMED_CONFLICT, CrisisType.BORDER_TENSION]:
            unit_type = "Peacekeeping Force"
        elif crisis.crisis_type in [CrisisType.REFUGEE_CAMP, CrisisType.HUMANITARIAN]:
            unit_type = "Humanitarian Aid"
        else:
            unit_type = "Diplomatic Mission"
        
        # Find optimal route from nearest major city
        major_cities = [name for name, loc in self.locations.items() 
                       if loc.is_city and loc.population > 1000000]
        
        best_route = []
        best_distance = float('inf')
        
        for city in major_cities:
            route, distance = self.pathfinder.find_path(city, crisis.location)
            if distance < best_distance:
                best_distance = distance
                best_route = route
        
        if best_route:
            response_unit = ResponseUnit(
                unit_type=unit_type,
                route=best_route,
                travel_time=best_distance / 50,  # Estimated travel time
                status="En Route"
            )
            
            self.agent.response_units.append(response_unit)
            crisis.response_sent = True
            
            self._add_message(f"📞 Support requested: {unit_type}", 
                            "success", f"Route: {' → '.join(best_route[:3])}{'...' if len(best_route) > 3 else ''}")
    
    def _update_crises(self):
        """Update crisis states"""
        for crisis in self.active_crises[:]:
            # Crises can escalate or de-escalate
            if random.random() < 0.01:
                crisis.severity = max(0.1, min(1.0, crisis.severity + random.uniform(-0.1, 0.1)))
                crisis.urgency = max(0.1, min(1.0, crisis.urgency + random.uniform(-0.05, 0.15)))
            
            # Remove resolved crises
            if crisis.response_sent and random.random() < 0.005:
                self.active_crises.remove(crisis)
                self._add_message(f"✅ Crisis resolved in {crisis.location}", 
                                "success", f"{crisis.crisis_type.value} - {crisis.affected_population:,} people helped")
    
    def _add_message(self, text: str, msg_type: str, details: str = ""):
        """Add message to communication queue"""
        self.message_queue.append({
            'text': text,
            'details': details,
            'type': msg_type,
            'time': time.time(),
            'life': 300  # 10 seconds at 30fps
        })
        
        # Keep only recent messages
        if len(self.message_queue) > 10:
            self.message_queue.pop(0)
    
    def _update_messages(self):
        """Update message queue"""
        for message in self.message_queue[:]:
            message['life'] -= 1
            if message['life'] <= 0:
                self.message_queue.remove(message)
    
    def render(self):
        """Render the complete system"""
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Render map
        self._render_map()
        
        # Render agent
        self._render_agent()
        
        # Render crises
        self._render_crises()
        
        # Render UI panels
        self._render_control_panel()
        self._render_communications()
        
        # Render title
        title = self.font_large.render("🚁 Mobile Crisis Response Agent - Real-time Operations", True, self.colors['text'])
        self.screen.blit(title, (20, 20))
        
        mission_info = self.font_medium.render(f"Mission Time: {self.agent.mission_time/60:.1f}min | Distance: {self.agent.total_distance_traveled:.0f}km | Status: {self.agent.status.value}", True, self.colors['text'])
        self.screen.blit(mission_info, (20, 50))
    
    def _render_map(self):
        """Render the map with roads and locations"""
        # Render roads
        for road in self.roads:
            start_loc = self.locations[road.start]
            end_loc = self.locations[road.end]
            
            # Road color based on condition and danger
            color_intensity = int(road.condition * 100 + 50)
            if road.danger_level > 0.6:
                color = (color_intensity, color_intensity//2, color_intensity//2)  # Reddish for dangerous
            else:
                color = (color_intensity, color_intensity, color_intensity)
            
            thickness = max(1, int(road.condition * 4))
            pygame.draw.line(self.screen, color, 
                           (start_loc.x, start_loc.y), (end_loc.x, end_loc.y), thickness)
        
        # Render locations
        for name, location in self.locations.items():
            # City size based on population
            if location.is_city:
                size = max(8, min(20, int(math.log10(location.population))))
                color = self.colors['city']
            else:
                size = 5
                color = (150, 150, 100)
            
            # Risk level ring
            if location.risk_level > 0.5:
                risk_size = size + int(location.risk_level * 10)
                risk_color = (255, int(255 * (1 - location.risk_level)), 0)
                pygame.draw.circle(self.screen, risk_color, (int(location.x), int(location.y)), risk_size, 2)
            
            pygame.draw.circle(self.screen, color, (int(location.x), int(location.y)), size)
            
            # Location name
            name_text = self.font_small.render(name, True, self.colors['text'])
            self.screen.blit(name_text, (location.x + size + 5, location.y - 8))
    
    def _render_agent(self):
        """Render the mobile agent"""
        # Agent position
        agent_x, agent_y = int(self.agent.x), int(self.agent.y)
        
        # Detection radius
        pygame.draw.circle(self.screen, (*self.colors['agent'][:3], 50), 
                         (agent_x, agent_y), int(self.agent.detection_radius), 1)
        
        # Agent body
        pygame.draw.circle(self.screen, self.colors['agent'], (agent_x, agent_y), 8)
        
        # Direction indicator
        if self.agent.current_path:
            next_idx = min(len(self.agent.current_path) - 1, int(self.agent.path_progress))
            if next_idx < len(self.agent.current_path):
                next_loc = self.locations[self.agent.current_path[next_idx]]
                dx = next_loc.x - self.agent.x
                dy = next_loc.y - self.agent.y
                if dx != 0 or dy != 0:
                    length = math.sqrt(dx*dx + dy*dy)
                    dx, dy = dx/length * 15, dy/length * 15
                    pygame.draw.line(self.screen, self.colors['agent'], 
                                   (agent_x, agent_y), (agent_x + dx, agent_y + dy), 3)
        
        # Agent label
        status_text = self.font_small.render(f"AGENT: {self.agent.status.value}", True, self.colors['agent'])
        self.screen.blit(status_text, (agent_x + 15, agent_y - 20))
        
        # Path visualization
        if self.agent.current_path:
            path_points = [(self.agent.x, self.agent.y)]
            for location_name in self.agent.current_path:
                loc = self.locations[location_name]
                path_points.append((loc.x, loc.y))
            
            if len(path_points) > 1:
                pygame.draw.lines(self.screen, (*self.colors['communication'], 100), False, path_points, 2)
    
    def _render_crises(self):
        """Render active crises"""
        for crisis in self.active_crises:
            x, y = crisis.coordinates
            
            # Crisis severity visualization
            size = max(5, int(crisis.severity * 15))
            pulse = (math.sin(self.animation_time * 4) + 1) * 0.3 + 0.7
            
            # Ensure pulse is valid
            pulse = max(0.1, min(2.0, pulse))
            
            if crisis.urgency > 0.7:
                base_color = self.colors['crisis_high']
            else:
                base_color = self.colors['crisis_low']
            
            # Safe color calculation
            try:
                color = tuple(max(0, min(255, int(c * pulse))) for c in base_color)
                # Ensure we have exactly 3 color components
                if len(color) != 3:
                    color = base_color
            except:
                color = base_color
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), size)
            
            # Crisis info
            crisis_text = f"{crisis.crisis_type.value[:15]}..."
            text_surface = self.font_small.render(crisis_text, True, color)
            self.screen.blit(text_surface, (x + size + 5, y - 8))
            
            # Affected population
            pop_text = f"{crisis.affected_population:,} affected"
            pop_surface = self.font_small.render(pop_text, True, (200, 200, 200))
            self.screen.blit(pop_surface, (x + size + 5, y + 5))
    
    def _render_control_panel(self):
        """Render agent control and status panel"""
        panel_x = 20
        panel_y = self.height - 200
        panel_width = 600
        panel_height = 180
        
        pygame.draw.rect(self.screen, self.colors['panel_bg'], 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['agent'], 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = self.font_medium.render("🎛️ Agent Control Center", True, self.colors['agent'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Agent stats
        stats_y = panel_y + 40
        stats = [
            f"Current Location: {self.agent.current_location}",
            f"Status: {self.agent.status.value}",
            f"Crises Discovered: {len(self.agent.crises_discovered)}",
            f"Response Units Deployed: {len(self.agent.response_units)}",
            f"Total Distance: {self.agent.total_distance_traveled:.0f}km",
        ]
        
        for i, stat in enumerate(stats):
            stat_text = self.font_small.render(stat, True, self.colors['text'])
            self.screen.blit(stat_text, (panel_x + 15, stats_y + i * 20))
        
        # Recent discoveries
        if self.agent.crises_discovered:
            recent_title = self.font_small.render("Recent Discoveries:", True, self.colors['text'])
            self.screen.blit(recent_title, (panel_x + 320, stats_y))
            
            for i, crisis in enumerate(self.agent.crises_discovered[-3:]):
                discovery_text = f"• {crisis.crisis_type.value} ({crisis.location})"
                discovery_surface = self.font_small.render(discovery_text, True, self.colors['crisis_high'])
                self.screen.blit(discovery_surface, (panel_x + 330, stats_y + 20 + i * 20))
    
    def _render_communications(self):
        """Render communications panel"""
        panel_x = self.width - 350
        panel_y = 100
        panel_width = 330
        panel_height = 400
        
        pygame.draw.rect(self.screen, self.colors['panel_bg'], 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['communication'], 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = self.font_medium.render("📡 Communications Log", True, self.colors['communication'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Messages
        msg_y = panel_y + 40
        for i, message in enumerate(self.message_queue[-15:]):  # Show last 15 messages
            # Message color based on type
            if message['type'] == 'alert':
                color = self.colors['crisis_high']
            elif message['type'] == 'warning':
                color = self.colors['crisis_low']
            elif message['type'] == 'success':
                color = (100, 255, 100)
            else:
                color = self.colors['text']
            
            # Fade older messages
            alpha = min(255, message['life'] * 2)
            fade_color = tuple(max(0, min(255, int(c * alpha / 255))) for c in color)
            
            # Message text
            msg_text = self.font_small.render(message['text'][:40] + ('...' if len(message['text']) > 40 else ''), 
                                            True, fade_color)
            self.screen.blit(msg_text, (panel_x + 10, msg_y + i * 22))
            
            # Details
            if message['details']:
                detail_text = self.font_small.render(f"  {message['details'][:35]}{'...' if len(message['details']) > 35 else ''}", 
                                                   True, tuple(max(0, min(255, int(c * 0.7 * alpha / 255))) for c in color))
                self.screen.blit(detail_text, (panel_x + 15, msg_y + i * 22 + 10))
    
    def run(self):
        """Run the mobile crisis response system"""
        running = True
        paused = False
        
        print("🚁 Mobile Crisis Response System Starting!")
        print("🌍 Agent will patrol Cameroon, DR Congo, Sudan")
        print("🎯 Watch real-time crisis discovery and response coordination")
        print("Controls: SPACE=pause, R=reset, ESC=exit")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("⏸️ Paused" if paused else "▶️ Resumed")
                    elif event.key == pygame.K_r:
                        print("🔄 Resetting mission...")
                        self.__init__(self.width, self.height)
                        
            if not paused:
                self.update_simulation()
                
            self.render()
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
            
        pygame.quit()
        print("🏆 Mission completed!")
        print(f"   Crises Discovered: {len(self.agent.crises_discovered)}")
        print(f"   Distance Traveled: {self.agent.total_distance_traveled:.0f}km")
        print(f"   Mission Duration: {self.agent.mission_time/60:.1f} minutes")
        print()
        input("Press ENTER to exit...")

if __name__ == "__main__":
    system = AfricaMobileSystem()
    system.run()