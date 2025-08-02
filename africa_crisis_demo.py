#!/usr/bin/env python3
"""
Africa Crisis Response AI - Focused Demo
Cameroon, DRC, and Sudan Crisis Management

A stunning demonstration showing AI decision-making for real African crisis scenarios.
Shows exactly WHY the agent makes each decision with clear visual feedback.
"""

import pygame
import numpy as np
import time
import random
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Crisis Types specific to these regions
class CrisisType(Enum):
    ETHNIC_TENSION = "Ethnic Tensions"
    RESOURCE_CONFLICT = "Resource Conflicts" 
    POLITICAL_INSTABILITY = "Political Instability"
    REFUGEE_CRISIS = "Refugee Crisis"
    ECONOMIC_COLLAPSE = "Economic Crisis"
    ARMED_GROUPS = "Armed Groups"
    BORDER_DISPUTE = "Border Disputes"

# Alert Levels
class AlertLevel(Enum):
    STABLE = 0      # Green
    WATCH = 1       # Yellow  
    WARNING = 2     # Orange
    CRISIS = 3      # Red
    EMERGENCY = 4   # Dark Red

# Available Actions
class ActionType(Enum):
    MONITOR = "Monitor Situation"
    DEPLOY_PEACEKEEPERS = "Deploy Peacekeepers"
    ECONOMIC_AID = "Economic Aid Package"
    DIPLOMATIC_INTERVENTION = "Diplomatic Intervention"
    HUMANITARIAN_AID = "Humanitarian Aid"
    EARLY_WARNING = "Early Warning System"
    REFUGEE_SUPPORT = "Refugee Support"
    MEDIATION = "Peace Mediation"

@dataclass
class Country:
    """Represents one of our target countries"""
    name: str
    capital: str
    population: int
    alert_level: AlertLevel
    
    # Stability indicators (0-1)
    political_stability: float
    economic_stability: float
    social_cohesion: float
    security_situation: float
    
    # Current crises
    active_crises: List[CrisisType]
    crisis_intensity: Dict[CrisisType, float]
    
    # Resources deployed
    peacekeepers: int
    aid_amount: float
    diplomatic_missions: int
    
    # Recent events
    recent_events: List[str]

class AgentDecision:
    """Represents an agent's decision with reasoning"""
    def __init__(self, country: str, action: ActionType, reasoning: List[str], confidence: float):
        self.country = country
        self.action = action
        self.reasoning = reasoning
        self.confidence = confidence
        self.timestamp = time.time()

class AfricaCrisisAI:
    """AI Crisis Response System for Cameroon, DRC, Sudan"""
    
    def __init__(self, width=1400, height=900):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🌍 Africa Crisis Response AI - Cameroon, DRC, Sudan")
        self.clock = pygame.time.Clock()
        
        # Enhanced color palette
        self.colors = {
            'background': (12, 20, 30),
            'panel_bg': (25, 35, 45),
            'text': (255, 255, 255),
            'text_dim': (180, 180, 180),
            'stable': (50, 200, 50),
            'watch': (255, 220, 50),
            'warning': (255, 150, 50),
            'crisis': (255, 80, 80),
            'emergency': (200, 30, 30),
            'agent': (50, 255, 150),
            'peacekeepers': (100, 150, 255),
            'aid': (255, 200, 100),
            'diplomacy': (150, 255, 150),
            'accent': (0, 200, 255),
            'gold': (255, 215, 0)
        }
        
        # Initialize countries with real data
        self.countries = self._initialize_countries()
        
        # Agent state
        self.current_focus = 0  # Currently focused country
        self.episode_reward = 0
        self.step_count = 0
        self.episode_count = 1
        self.total_crises_prevented = 0
        self.total_lives_saved = 0
        
        # Decision tracking
        self.current_decision = None
        self.decision_history = []
        self.thinking_display = []
        
        # Animation
        self.animation_time = 0
        self.particles = []
        self.alert_pulses = {}
        
        # Fonts
        self.font_title = pygame.font.Font(None, 40)
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Real-world crisis scenarios
        self._initialize_crisis_scenarios()
        
    def _initialize_countries(self) -> List[Country]:
        """Initialize the three target countries with realistic data"""
        countries = [
            Country(
                name="Cameroon",
                capital="Yaoundé", 
                population=27_000_000,
                alert_level=AlertLevel.WARNING,
                political_stability=0.4,
                economic_stability=0.5,
                social_cohesion=0.3,
                security_situation=0.4,
                active_crises=[CrisisType.ETHNIC_TENSION, CrisisType.ARMED_GROUPS],
                crisis_intensity={
                    CrisisType.ETHNIC_TENSION: 0.7,
                    CrisisType.ARMED_GROUPS: 0.6
                },
                peacekeepers=0,
                aid_amount=0,
                diplomatic_missions=0,
                recent_events=["Anglophone separatist activity", "Boko Haram incursions"]
            ),
            Country(
                name="DR Congo",
                capital="Kinshasa",
                population=95_000_000,
                alert_level=AlertLevel.CRISIS,
                political_stability=0.2,
                economic_stability=0.3,
                social_cohesion=0.3,
                security_situation=0.2,
                active_crises=[CrisisType.ARMED_GROUPS, CrisisType.RESOURCE_CONFLICT, CrisisType.REFUGEE_CRISIS],
                crisis_intensity={
                    CrisisType.ARMED_GROUPS: 0.9,
                    CrisisType.RESOURCE_CONFLICT: 0.8,
                    CrisisType.REFUGEE_CRISIS: 0.7
                },
                peacekeepers=15000,
                aid_amount=500_000_000,
                diplomatic_missions=3,
                recent_events=["M23 rebel activity", "Mining conflicts in Kivu", "Mass displacement"]
            ),
            Country(
                name="Sudan",
                capital="Khartoum",
                population=45_000_000,
                alert_level=AlertLevel.EMERGENCY,
                political_stability=0.1,
                economic_stability=0.2,
                social_cohesion=0.2,
                security_situation=0.1,
                active_crises=[CrisisType.POLITICAL_INSTABILITY, CrisisType.ARMED_GROUPS, CrisisType.ECONOMIC_COLLAPSE],
                crisis_intensity={
                    CrisisType.POLITICAL_INSTABILITY: 0.95,
                    CrisisType.ARMED_GROUPS: 0.9,
                    CrisisType.ECONOMIC_COLLAPSE: 0.8
                },
                peacekeepers=8000,
                aid_amount=1_200_000_000,
                diplomatic_missions=5,
                recent_events=["SAF vs RSF conflict", "Humanitarian crisis", "Economic collapse"]
            )
        ]
        return countries
        
    def _initialize_crisis_scenarios(self):
        """Initialize realistic crisis scenarios for each country"""
        self.crisis_scenarios = {
            "Cameroon": [
                "Separatist violence escalates in Anglophone regions",
                "Boko Haram attacks increase in Far North",
                "Inter-community clashes over land resources",
                "Political tensions rise ahead of elections"
            ],
            "DR Congo": [
                "M23 rebels advance toward Goma",
                "New armed group emerges in South Kivu", 
                "Massive displacement from mining conflicts",
                "Ebola outbreak reported in eastern regions"
            ],
            "Sudan": [
                "Fighting intensifies in Khartoum",
                "RSF forces advance on strategic positions",
                "Humanitarian corridors come under attack",
                "Economic collapse triggers mass protests"
            ]
        }
        
    def update_simulation(self):
        """Update the entire simulation"""
        self.step_count += 1
        self.animation_time += 0.05
        
        # Agent decision-making every 2 seconds (60 frames)
        if self.step_count % 60 == 0:
            self._agent_decision_cycle()
            
        # Random crisis events
        if random.random() < 0.015:  # 1.5% chance per frame
            self._trigger_crisis_event()
            
        # Update countries
        for country in self.countries:
            self._update_country(country)
            
        # Update visual effects
        self._update_particles()
        self._update_alert_pulses()
        
    def _agent_decision_cycle(self):
        """Complete AI decision-making cycle with visible reasoning"""
        # Clear previous thinking
        self.thinking_display = []
        
        # Step 1: Analyze all countries
        self.thinking_display.append("🔍 ANALYZING SITUATION...")
        
        # Step 2: Assess threats
        threat_analysis = {}
        for i, country in enumerate(self.countries):
            threat_score = self._calculate_threat_score(country)
            threat_analysis[country.name] = threat_score
            
        # Step 3: Determine priority
        priority_country = max(threat_analysis.items(), key=lambda x: x[1])
        self.current_focus = next(i for i, c in enumerate(self.countries) if c.name == priority_country[0])
        
        self.thinking_display.append(f"🎯 PRIORITY: {priority_country[0]}")
        self.thinking_display.append(f"⚠️  THREAT LEVEL: {priority_country[1]:.1f}/10")
        
        # Step 4: Choose action
        target_country = self.countries[self.current_focus]
        action, reasoning = self._choose_action(target_country)
        
        # Step 5: Execute action
        self._execute_action(target_country, action)
        
        # Store decision
        confidence = min(1.0, priority_country[1] / 10.0)
        self.current_decision = AgentDecision(
            target_country.name, action, reasoning, confidence
        )
        self.decision_history.append(self.current_decision)
        
        # Keep only recent decisions
        if len(self.decision_history) > 10:
            self.decision_history.pop(0)
            
    def _calculate_threat_score(self, country: Country) -> float:
        """Calculate threat score for prioritization (0-10 scale)"""
        # Base threat from alert level
        base_threat = country.alert_level.value * 2
        
        # Instability factors
        instability = (
            (1 - country.political_stability) * 2 +
            (1 - country.security_situation) * 2 +
            (1 - country.social_cohesion) * 1
        )
        
        # Crisis intensity
        crisis_factor = sum(country.crisis_intensity.values()) / len(country.crisis_intensity) if country.crisis_intensity else 0
        
        # Population impact multiplier
        population_factor = math.log10(country.population / 1_000_000) * 0.5
        
        total_threat = base_threat + instability + crisis_factor + population_factor
        return min(10.0, total_threat)
        
    def _choose_action(self, country: Country) -> Tuple[ActionType, List[str]]:
        """Choose the best action with reasoning"""
        reasoning = []
        
        # Analyze country situation
        if country.alert_level == AlertLevel.EMERGENCY:
            reasoning.append("🚨 EMERGENCY: Immediate intervention required")
            if country.security_situation < 0.3:
                reasoning.append("🛡️  Security collapse - deploying peacekeepers")
                return ActionType.DEPLOY_PEACEKEEPERS, reasoning
            else:
                reasoning.append("🏥 Humanitarian crisis - sending aid")
                return ActionType.HUMANITARIAN_AID, reasoning
                
        elif country.alert_level == AlertLevel.CRISIS:
            reasoning.append("🔴 CRISIS: Active intervention needed")
            if CrisisType.ARMED_GROUPS in country.active_crises:
                reasoning.append("⚔️  Armed conflict active - military response")
                return ActionType.DEPLOY_PEACEKEEPERS, reasoning
            elif country.economic_stability < 0.3:
                reasoning.append("💰 Economic collapse - financial support")
                return ActionType.ECONOMIC_AID, reasoning
            else:
                reasoning.append("🤝 Diplomatic solution possible")
                return ActionType.DIPLOMATIC_INTERVENTION, reasoning
                
        elif country.alert_level == AlertLevel.WARNING:
            reasoning.append("🟠 WARNING: Preventive action recommended")
            if country.political_stability < 0.4:
                reasoning.append("🏛️  Political instability - diplomatic engagement")
                return ActionType.DIPLOMATIC_INTERVENTION, reasoning
            else:
                reasoning.append("📊 Early warning system activation")
                return ActionType.EARLY_WARNING, reasoning
                
        else:
            reasoning.append("📡 Situation stable - continuing monitoring")
            return ActionType.MONITOR, reasoning
            
    def _execute_action(self, country: Country, action: ActionType):
        """Execute the chosen action and update country state"""
        if action == ActionType.DEPLOY_PEACEKEEPERS:
            country.peacekeepers += 2000
            country.security_situation = min(1.0, country.security_situation + 0.15)
            self.episode_reward += 50
            self.total_lives_saved += random.randint(100, 500)
            self._add_particle(country.name, "🛡️ Peacekeepers Deployed", self.colors['peacekeepers'])
            
        elif action == ActionType.ECONOMIC_AID:
            country.aid_amount += 100_000_000
            country.economic_stability = min(1.0, country.economic_stability + 0.2)
            self.episode_reward += 30
            self._add_particle(country.name, "💰 Aid Package Sent", self.colors['aid'])
            
        elif action == ActionType.DIPLOMATIC_INTERVENTION:
            country.diplomatic_missions += 1
            country.political_stability = min(1.0, country.political_stability + 0.25)
            self.episode_reward += 40
            self._add_particle(country.name, "🤝 Diplomatic Mission", self.colors['diplomacy'])
            
        elif action == ActionType.HUMANITARIAN_AID:
            country.aid_amount += 50_000_000
            country.social_cohesion = min(1.0, country.social_cohesion + 0.1)
            self.episode_reward += 25
            self.total_lives_saved += random.randint(200, 800)
            self._add_particle(country.name, "🏥 Humanitarian Aid", self.colors['aid'])
            
        elif action == ActionType.EARLY_WARNING:
            # Prevents crisis escalation
            for crisis_type in country.active_crises:
                country.crisis_intensity[crisis_type] *= 0.9
            self.episode_reward += 15
            self.total_crises_prevented += 1
            self._add_particle(country.name, "⚠️ Early Warning", self.colors['accent'])
            
        elif action == ActionType.MONITOR:
            self.episode_reward += 5
            self._add_particle(country.name, "📡 Monitoring", self.colors['text_dim'])
            
    def _trigger_crisis_event(self):
        """Trigger a realistic crisis event"""
        country = random.choice(self.countries)
        scenarios = self.crisis_scenarios.get(country.name, [])
        
        if scenarios:
            event = random.choice(scenarios)
            country.recent_events.append(event)
            
            # Keep only recent events
            if len(country.recent_events) > 3:
                country.recent_events.pop(0)
                
            # Escalate situation
            if country.alert_level.value < 4:
                old_level = country.alert_level.value
                country.alert_level = AlertLevel(min(4, old_level + 1))
                
            # Worsen conditions
            country.political_stability = max(0, country.political_stability - 0.1)
            country.security_situation = max(0, country.security_situation - 0.15)
            
            self.episode_reward -= 100
            self._add_particle(country.name, f"🚨 CRISIS: {event[:20]}...", self.colors['emergency'])
            
    def _update_country(self, country: Country):
        """Update country state over time"""
        # Natural fluctuations
        country.political_stability += random.uniform(-0.01, 0.005)
        country.economic_stability += random.uniform(-0.01, 0.005)
        country.social_cohesion += random.uniform(-0.01, 0.005)
        country.security_situation += random.uniform(-0.01, 0.005)
        
        # Clamp values
        for attr in ['political_stability', 'economic_stability', 'social_cohesion', 'security_situation']:
            setattr(country, attr, max(0, min(1, getattr(country, attr))))
            
        # Update alert level based on overall stability
        avg_stability = (
            country.political_stability + 
            country.economic_stability + 
            country.social_cohesion + 
            country.security_situation
        ) / 4
        
        if avg_stability > 0.8:
            country.alert_level = AlertLevel.STABLE
        elif avg_stability > 0.6:
            country.alert_level = AlertLevel.WATCH
        elif avg_stability > 0.4:
            country.alert_level = AlertLevel.WARNING
        elif avg_stability > 0.2:
            country.alert_level = AlertLevel.CRISIS
        else:
            country.alert_level = AlertLevel.EMERGENCY
            
    def _add_particle(self, country_name: str, text: str, color: Tuple[int, int, int]):
        """Add visual effect particle"""
        country_idx = next(i for i, c in enumerate(self.countries) if c.name == country_name)
        x = 50 + country_idx * 420 + random.randint(-30, 30)
        y = 200 + random.randint(-20, 20)
        
        self.particles.append({
            'x': x,
            'y': y,
            'text': text,
            'color': color,
            'life': 120,  # 4 seconds at 30fps
            'vy': -1,
            'alpha': 255
        })
        
    def _update_particles(self):
        """Update visual effect particles"""
        for particle in self.particles[:]:
            particle['life'] -= 1
            particle['y'] += particle['vy']
            particle['alpha'] = int(255 * (particle['life'] / 120))
            
            if particle['life'] <= 0:
                self.particles.remove(particle)
                
    def _update_alert_pulses(self):
        """Update alert pulse animations"""
        for i, country in enumerate(self.countries):
            if country.alert_level.value >= 3:  # Crisis or Emergency
                if country.name not in self.alert_pulses:
                    self.alert_pulses[country.name] = 0
                self.alert_pulses[country.name] += 0.1
            elif country.name in self.alert_pulses:
                del self.alert_pulses[country.name]
                
    def render(self):
        """Render the complete demonstration"""
        # Clear screen with gradient background
        self._render_background()
        
        # Title and main stats
        self._render_header()
        
        # Country panels
        self._render_countries()
        
        # Agent decision panel
        self._render_agent_panel()
        
        # Real-time thinking display
        self._render_thinking_panel()
        
        # Render particles
        self._render_particles()
        
        # Instructions
        self._render_instructions()
        
    def _render_background(self):
        """Render gradient background"""
        for y in range(self.height):
            color_ratio = y / self.height
            r = int(self.colors['background'][0] * (1 - color_ratio * 0.3))
            g = int(self.colors['background'][1] * (1 - color_ratio * 0.3))
            b = int(self.colors['background'][2] * (1 - color_ratio * 0.3))
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.width, y))
            
    def _render_header(self):
        """Render main header with statistics"""
        # Main title
        title = self.font_title.render("🌍 Africa Crisis Response AI", True, self.colors['gold'])
        self.screen.blit(title, (20, 20))
        
        # Subtitle
        subtitle = self.font_medium.render("Real-time AI Decision Making: Cameroon • DR Congo • Sudan", True, self.colors['accent'])
        self.screen.blit(subtitle, (20, 60))
        
        # Episode stats
        stats_text = f"Episode {self.episode_count} | Step {self.step_count} | Reward: {self.episode_reward:.0f} | Lives Saved: {self.total_lives_saved:,} | Crises Prevented: {self.total_crises_prevented}"
        stats = self.font_small.render(stats_text, True, self.colors['text'])
        self.screen.blit(stats, (20, 85))
        
    def _render_countries(self):
        """Render the three country panels"""
        for i, country in enumerate(self.countries):
            self._render_country_panel(country, i)
            
    def _render_country_panel(self, country: Country, index: int):
        """Render detailed country panel"""
        x = 50 + index * 420
        y = 120
        width = 380
        height = 400
        
        # Alert level colors
        alert_colors = {
            AlertLevel.STABLE: self.colors['stable'],
            AlertLevel.WATCH: self.colors['watch'],
            AlertLevel.WARNING: self.colors['warning'],
            AlertLevel.CRISIS: self.colors['crisis'],
            AlertLevel.EMERGENCY: self.colors['emergency']
        }
        
        border_color = alert_colors[country.alert_level]
        
        # Pulsing effect for high alerts
        if country.name in self.alert_pulses:
            pulse = (math.sin(self.alert_pulses[country.name] * 3) + 1) * 0.2
            border_color = tuple(max(0, min(255, int(c * (1 + pulse)))) for c in border_color)
            
        # Panel background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (x, y, width, height))
        pygame.draw.rect(self.screen, border_color, (x, y, width, height), 4)
        
        # Agent focus indicator
        if index == self.current_focus:
            pygame.draw.rect(self.screen, self.colors['agent'], (x-2, y-2, width+4, height+4), 2)
            focus_text = self.font_small.render("🎯 AI FOCUSED HERE", True, self.colors['agent'])
            self.screen.blit(focus_text, (x + 10, y + 10))
            
        # Country header
        header_y = y + (35 if index == self.current_focus else 15)
        country_name = self.font_large.render(f"🏴 {country.name}", True, self.colors['text'])
        self.screen.blit(country_name, (x + 15, header_y))
        
        capital_pop = self.font_small.render(f"Capital: {country.capital} | Pop: {country.population:,}", True, self.colors['text_dim'])
        self.screen.blit(capital_pop, (x + 15, header_y + 30))
        
        # Alert level
        alert_text = f"🚨 ALERT: {country.alert_level.name}"
        alert = self.font_medium.render(alert_text, True, border_color)
        self.screen.blit(alert, (x + 15, header_y + 55))
        
        # Stability indicators
        stability_y = header_y + 85
        self._render_stability_bars(x + 15, stability_y, country)
        
        # Active crises
        crises_y = stability_y + 120
        crises_title = self.font_medium.render("🔥 Active Crises:", True, self.colors['text'])
        self.screen.blit(crises_title, (x + 15, crises_y))
        
        for i, crisis in enumerate(list(country.active_crises)[:3]):
            intensity = country.crisis_intensity.get(crisis, 0)
            crisis_text = f"• {crisis.value} ({intensity:.0%})"
            crisis_color = self.colors['crisis'] if intensity > 0.6 else self.colors['warning']
            crisis_render = self.font_small.render(crisis_text, True, crisis_color)
            self.screen.blit(crisis_render, (x + 25, crises_y + 25 + i * 20))
            
        # Resources deployed
        resources_y = crises_y + 90
        resources_title = self.font_medium.render("🛡️ Resources Deployed:", True, self.colors['text'])
        self.screen.blit(resources_title, (x + 15, resources_y))
        
        if country.peacekeepers > 0:
            pk_text = f"👥 Peacekeepers: {country.peacekeepers:,}"
            pk_render = self.font_small.render(pk_text, True, self.colors['peacekeepers'])
            self.screen.blit(pk_render, (x + 25, resources_y + 25))
            
        if country.aid_amount > 0:
            aid_text = f"💰 Aid: ${country.aid_amount/1_000_000:.0f}M"
            aid_render = self.font_small.render(aid_text, True, self.colors['aid'])
            self.screen.blit(aid_render, (x + 25, resources_y + 45))
            
        if country.diplomatic_missions > 0:
            diplo_text = f"🤝 Diplomatic Missions: {country.diplomatic_missions}"
            diplo_render = self.font_small.render(diplo_text, True, self.colors['diplomacy'])
            self.screen.blit(diplo_render, (x + 25, resources_y + 65))
            
        # Recent events
        events_y = resources_y + 90
        if country.recent_events:
            events_title = self.font_small.render("📰 Recent Events:", True, self.colors['text_dim'])
            self.screen.blit(events_title, (x + 15, events_y))
            
            for i, event in enumerate(country.recent_events[-2:]):  # Show last 2 events
                event_text = f"• {event[:35]}..."
                event_render = self.font_small.render(event_text, True, self.colors['text_dim'])
                self.screen.blit(event_render, (x + 25, events_y + 20 + i * 15))
                
    def _render_stability_bars(self, x: int, y: int, country: Country):
        """Render stability indicator bars"""
        indicators = [
            ("Political", country.political_stability),
            ("Economic", country.economic_stability), 
            ("Social", country.social_cohesion),
            ("Security", country.security_situation)
        ]
        
        for i, (label, value) in enumerate(indicators):
            bar_y = y + i * 25
            
            # Label
            label_text = self.font_small.render(f"{label}:", True, self.colors['text'])
            self.screen.blit(label_text, (x, bar_y))
            
            # Background bar
            bar_x = x + 80
            bar_width = 200
            bar_height = 12
            pygame.draw.rect(self.screen, (40, 40, 40), (bar_x, bar_y, bar_width, bar_height))
            
            # Value bar
            fill_width = int(bar_width * value)
            if value > 0.7:
                color = self.colors['stable']
            elif value > 0.4:
                color = self.colors['warning']
            else:
                color = self.colors['crisis']
                
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height))
            
            # Value text
            value_text = f"{value:.0%}"
            value_render = self.font_small.render(value_text, True, self.colors['text'])
            self.screen.blit(value_render, (bar_x + bar_width + 10, bar_y - 2))
            
    def _render_agent_panel(self):
        """Render AI agent decision panel"""
        panel_x = 50
        panel_y = 540
        panel_width = 650
        panel_height = 280
        
        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['agent'], (panel_x, panel_y, panel_width, panel_height), 3)
        
        # Title
        title = self.font_large.render("🧠 AI Agent Decision Center", True, self.colors['agent'])
        self.screen.blit(title, (panel_x + 20, panel_y + 15))
        
        # Current decision
        if self.current_decision:
            decision_y = panel_y + 50
            
            # Action
            action_text = f"🎯 Action: {self.current_decision.action.value}"
            action_render = self.font_medium.render(action_text, True, self.colors['text'])
            self.screen.blit(action_render, (panel_x + 20, decision_y))
            
            # Target
            target_text = f"📍 Target: {self.current_decision.country}"
            target_render = self.font_medium.render(target_text, True, self.colors['accent'])
            self.screen.blit(target_render, (panel_x + 20, decision_y + 30))
            
            # Confidence
            confidence_text = f"🎲 Confidence: {self.current_decision.confidence:.0%}"
            confidence_render = self.font_medium.render(confidence_text, True, self.colors['gold'])
            self.screen.blit(confidence_render, (panel_x + 20, decision_y + 60))
            
            # Reasoning
            reasoning_title = self.font_medium.render("💭 AI Reasoning:", True, self.colors['text'])
            self.screen.blit(reasoning_title, (panel_x + 20, decision_y + 100))
            
            for i, reason in enumerate(self.current_decision.reasoning[:4]):  # Show up to 4 reasons
                reason_render = self.font_small.render(f"  {reason}", True, self.colors['text_dim'])
                self.screen.blit(reason_render, (panel_x + 30, decision_y + 130 + i * 20))
        else:
            no_decision = self.font_medium.render("🔄 AI analyzing situation...", True, self.colors['text_dim'])
            self.screen.blit(no_decision, (panel_x + 20, panel_y + 50))
            
    def _render_thinking_panel(self):
        """Render real-time AI thinking display"""
        panel_x = 720
        panel_y = 540
        panel_width = 630
        panel_height = 280
        
        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['accent'], (panel_x, panel_y, panel_width, panel_height), 3)
        
        # Title
        title = self.font_large.render("⚡ Live AI Thinking Process", True, self.colors['accent'])
        self.screen.blit(title, (panel_x + 20, panel_y + 15))
        
        # Current thinking
        thinking_y = panel_y + 50
        for i, thought in enumerate(self.thinking_display[:8]):  # Show up to 8 thoughts
            alpha = 255 - (i * 20)  # Fade older thoughts
            color = tuple(max(50, c - i * 15) for c in self.colors['text'])
            
            thought_render = self.font_small.render(thought, True, color)
            self.screen.blit(thought_render, (panel_x + 20, thinking_y + i * 20))
            
        # Decision history
        history_title = self.font_medium.render("📊 Recent Decisions:", True, self.colors['text'])
        self.screen.blit(history_title, (panel_x + 20, panel_y + 220))
        
        recent_decisions = self.decision_history[-3:]  # Last 3 decisions
        for i, decision in enumerate(recent_decisions):
            time_ago = int(time.time() - decision.timestamp)
            decision_text = f"{decision.country}: {decision.action.value.split()[0]} ({time_ago}s ago)"
            decision_render = self.font_small.render(decision_text, True, self.colors['text_dim'])
            self.screen.blit(decision_render, (panel_x + 30, panel_y + 245 + i * 15))
            
    def _render_particles(self):
        """Render visual effect particles"""
        for particle in self.particles:
            alpha_color = tuple(max(0, min(255, int(c * particle['alpha'] / 255))) for c in particle['color'])
            if particle['alpha'] > 50:  # Only render visible particles
                text_surface = self.font_small.render(particle['text'], True, alpha_color)
                self.screen.blit(text_surface, (particle['x'], particle['y']))
                
    def _render_instructions(self):
        """Render control instructions"""
        instructions = [
            "🎮 Controls: SPACE=Pause | R=Reset | ESC=Exit",
            "👀 Watch the AI analyze threats, make decisions, and explain its reasoning!",
            "🎯 The AI focuses on the highest threat country and takes appropriate action."
        ]
        
        for i, instruction in enumerate(instructions):
            color = self.colors['text_dim'] if i > 0 else self.colors['text']
            inst_render = self.font_small.render(instruction, True, color)
            self.screen.blit(inst_render, (20, self.height - 60 + i * 20))
            
    def run(self):
        """Run the demonstration"""
        running = True
        paused = False
        
        print("🌍 Africa Crisis Response AI Demo Starting!")
        print("🎯 Monitoring: Cameroon, DR Congo, Sudan")
        print("🧠 Watch AI decision-making in real-time!")
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
                        print("🔄 Resetting simulation...")
                        self.__init__(self.width, self.height)
                        
            if not paused:
                self.update_simulation()
                
                # Reset episode periodically
                if self.step_count >= 600:  # 20 seconds at 30fps
                    self.episode_count += 1
                    self.step_count = 0
                    self.episode_reward = 0
                    print(f"🎬 Episode {self.episode_count} started! Crises prevented: {self.total_crises_prevented}, Lives saved: {self.total_lives_saved:,}")
                    
            self.render()
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
            
        pygame.quit()
        print("🏆 Demo completed! Final stats:")
        print(f"   Episodes: {self.episode_count}")
        print(f"   Crises Prevented: {self.total_crises_prevented}")
        print(f"   Lives Saved: {self.total_lives_saved:,}")
        print()
        input("Press ENTER to exit...")

if __name__ == "__main__":
    demo = AfricaCrisisAI()
    demo.run()