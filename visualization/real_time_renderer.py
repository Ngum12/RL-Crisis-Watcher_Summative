"""
Advanced Real-Time Renderer for Crisis Response Environment

This module provides sophisticated 2D/3D visualization capabilities including:
- Interactive real-time environment rendering
- Advanced visual effects and animations
- Multi-layered information display
- Performance monitoring overlays
- Agent behavior visualization
- Crisis escalation animations

Features:
- High-performance Pygame rendering
- Smooth animations and transitions
- Interactive zoom and pan
- Real-time data overlays
- Professional color schemes
- Particle effects for crisis events
"""

import pygame
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import threading
import time
from collections import deque

class RenderMode(Enum):
    """Rendering modes for different visualization needs"""
    OVERVIEW = "overview"           # Full world view
    DETAILED = "detailed"          # Focused region view
    METRICS = "metrics"            # Performance dashboard
    TRAINING = "training"          # Training progress view
    COMPARISON = "comparison"      # Algorithm comparison

@dataclass
class VisualEffect:
    """Visual effect for animations"""
    x: float
    y: float
    duration: float
    elapsed: float
    effect_type: str
    intensity: float
    color: Tuple[int, int, int]
    
class ParticleSystem:
    """Advanced particle system for visual effects"""
    
    def __init__(self):
        self.particles = []
        self.max_particles = 1000
    
    def add_explosion(self, x: float, y: float, color: Tuple[int, int, int], intensity: float = 1.0):
        """Add explosion effect at position"""
        num_particles = int(20 * intensity)
        for _ in range(num_particles):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(2, 8) * intensity
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': 1.0,
                'decay': np.random.uniform(0.01, 0.03),
                'color': color,
                'size': np.random.uniform(2, 6)
            })
    
    def update(self, dt: float):
        """Update particle system"""
        for particle in self.particles[:]:
            particle['x'] += particle['vx'] * dt
            particle['y'] += particle['vy'] * dt
            particle['life'] -= particle['decay'] * dt
            particle['size'] *= 0.99
            
            if particle['life'] <= 0 or particle['size'] < 0.5:
                self.particles.remove(particle)
    
    def render(self, screen: pygame.Surface):
        """Render all particles"""
        for particle in self.particles:
            alpha = int(255 * particle['life'])
            color = (*particle['color'], alpha)
            size = max(1, int(particle['size']))
            
            # Create surface with per-pixel alpha
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (size, size), size)
            screen.blit(surf, (particle['x'] - size, particle['y'] - size))

class CrisisRenderer:
    """
    Advanced renderer for the Crisis Response environment
    
    Provides sophisticated visualization with:
    - Interactive world map display
    - Real-time crisis indicators
    - Resource deployment visualization
    - Agent action feedback
    - Performance metrics overlay
    - Training progress monitoring
    """
    
    def __init__(self, 
                 width: int = 1400, 
                 height: int = 900,
                 fullscreen: bool = False,
                 anti_alias: bool = True):
        
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        
        # Screen setup
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE
        if anti_alias:
            flags |= pygame.SCALED
        if fullscreen:
            flags |= pygame.FULLSCREEN
            
        self.screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption("Crisis Response AI - Advanced Visualization")
        
        # Rendering components
        self.clock = pygame.time.Clock()
        self.particle_system = ParticleSystem()
        self.visual_effects = []
        
        # Fonts
        self.fonts = {
            'title': pygame.font.Font(None, 36),
            'large': pygame.font.Font(None, 28),
            'medium': pygame.font.Font(None, 24),
            'small': pygame.font.Font(None, 18),
            'tiny': pygame.font.Font(None, 14)
        }
        
        # Professional color palette
        self.colors = {
            # Background and UI
            'bg_primary': (15, 25, 35),
            'bg_secondary': (25, 35, 45),
            'bg_tertiary': (35, 45, 55),
            'ui_border': (70, 80, 90),
            'ui_highlight': (100, 150, 200),
            
            # Text colors
            'text_primary': (255, 255, 255),
            'text_secondary': (200, 200, 200),
            'text_accent': (100, 200, 255),
            'text_warning': (255, 200, 100),
            'text_danger': (255, 100, 100),
            
            # Alert levels
            'alert_green': (50, 200, 50),
            'alert_yellow': (255, 220, 50),
            'alert_orange': (255, 150, 50),
            'alert_red': (255, 80, 80),
            'alert_critical': (200, 30, 30),
            
            # Resources
            'peacekeepers': (100, 150, 255),
            'humanitarian': (255, 200, 100),
            'diplomatic': (150, 255, 150),
            'economic': (255, 150, 255),
            
            # Effects
            'explosion': (255, 100, 50),
            'success': (100, 255, 100),
            'warning': (255, 255, 100),
            'info': (100, 200, 255)
        }
        
        # Rendering state
        self.mode = RenderMode.OVERVIEW
        self.zoom_level = 1.0
        self.camera_x = 0
        self.camera_y = 0
        self.selected_region = None
        self.animation_time = 0.0
        
        # Performance tracking
        self.fps_history = deque(maxlen=60)
        self.frame_times = deque(maxlen=60)
        self.last_frame_time = time.time()
        
        # UI state
        self.show_debug = False
        self.show_metrics = True
        self.show_grid = True
        self.smooth_animations = True
        
        # World map layout (4x3 grid of regions)
        self.world_grid = {
            'cols': 4,
            'rows': 3,
            'margin': 20,
            'padding': 10
        }
        
    def calculate_region_bounds(self, region_id: int) -> Tuple[int, int, int, int]:
        """Calculate screen bounds for a region"""
        cols = self.world_grid['cols']
        rows = self.world_grid['rows']
        margin = self.world_grid['margin']
        padding = self.world_grid['padding']
        
        # Calculate available space
        available_width = self.width - (2 * margin) - 300  # Leave space for sidebar
        available_height = self.height - (2 * margin) - 150  # Leave space for bottom panel
        
        cell_width = (available_width - (cols - 1) * padding) // cols
        cell_height = (available_height - (rows - 1) * padding) // rows
        
        row = region_id // cols
        col = region_id % cols
        
        x = margin + col * (cell_width + padding)
        y = margin + row * (cell_height + padding)
        
        return x, y, cell_width, cell_height
    
    def render_world_map(self, environment):
        """Render the main world map with all regions"""
        # Background
        self.screen.fill(self.colors['bg_primary'])
        
        # Draw grid background
        if self.show_grid:
            self._draw_background_grid()
        
        # Render each region
        for i, region in enumerate(environment.regions):
            self._render_region(region, i, environment)
        
        # Render connections between regions
        self._render_region_connections(environment)
        
    def _draw_background_grid(self):
        """Draw subtle background grid"""
        grid_color = (30, 40, 50)
        spacing = 50
        
        for x in range(0, self.width, spacing):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, spacing):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.width, y))
    
    def _render_region(self, region, region_id: int, environment):
        """Render a single region with advanced visual elements"""
        x, y, width, height = self.calculate_region_bounds(region_id)
        
        # Calculate region state
        conflict_prob = environment._calculate_conflict_probability(region)
        alert_level = region.current_alert_level
        
        # Determine region color with smooth transitions
        alert_colors = {
            0: self.colors['alert_green'],      # GREEN
            1: self.colors['alert_yellow'],     # YELLOW
            2: self.colors['alert_orange'],     # ORANGE
            3: self.colors['alert_red'],        # RED
            4: self.colors['alert_critical']    # CRITICAL
        }
        
        base_color = alert_colors[alert_level.value]
        
        # Add pulsing effect for high-risk regions
        if alert_level.value >= 3:
            pulse = (math.sin(self.animation_time * 3) + 1) * 0.1
            base_color = tuple(min(255, int(c * (1 + pulse))) for c in base_color)
        
        # Draw main region rectangle with gradient effect
        self._draw_gradient_rect(x, y, width, height, base_color)
        
        # Draw border with thickness based on alert level
        border_thickness = max(2, alert_level.value)
        pygame.draw.rect(self.screen, self.colors['ui_border'], 
                        (x, y, width, height), border_thickness)
        
        # Region header
        self._render_region_header(region, x, y, width)
        
        # Stability indicators
        self._render_stability_bars(region, x, y + 40, width)
        
        # Resource deployment visualization
        self._render_resource_deployment(region, x, y + height - 80, width, 30)
        
        # Active conflicts indicators
        self._render_active_conflicts(region, x + width - 60, y + 5)
        
        # Interaction highlight
        if self.selected_region == region_id:
            pygame.draw.rect(self.screen, self.colors['ui_highlight'], 
                           (x - 3, y - 3, width + 6, height + 6), 3)
    
    def _draw_gradient_rect(self, x: int, y: int, width: int, height: int, 
                           color: Tuple[int, int, int]):
        """Draw rectangle with gradient effect"""
        # Create surface for gradient
        surf = pygame.Surface((width, height))
        
        # Simple vertical gradient
        for i in range(height):
            factor = 1.0 - (i / height) * 0.3  # Darker at bottom
            grad_color = tuple(int(c * factor) for c in color)
            pygame.draw.line(surf, grad_color, (0, i), (width, i))
        
        self.screen.blit(surf, (x, y))
    
    def _render_region_header(self, region, x: int, y: int, width: int):
        """Render region name and alert level"""
        # Region name
        name_surface = self.fonts['medium'].render(region.name, True, self.colors['text_primary'])
        name_rect = name_surface.get_rect()
        name_rect.centerx = x + width // 2
        name_rect.y = y + 5
        self.screen.blit(name_surface, name_rect)
        
        # Alert level
        alert_text = f"Alert: {region.current_alert_level.name}"
        alert_color = self.colors['text_warning'] if region.current_alert_level.value >= 2 else self.colors['text_secondary']
        alert_surface = self.fonts['small'].render(alert_text, True, alert_color)
        alert_rect = alert_surface.get_rect()
        alert_rect.centerx = x + width // 2
        alert_rect.y = y + 25
        self.screen.blit(alert_surface, alert_rect)
    
    def _render_stability_bars(self, region, x: int, y: int, width: int):
        """Render stability indicators as horizontal bars"""
        bar_height = 8
        bar_spacing = 12
        bar_width = width - 20
        
        indicators = [
            ("Economic", region.economic_stability, self.colors['economic']),
            ("Political", region.political_stability, self.colors['diplomatic']),
            ("Social", region.social_cohesion, self.colors['humanitarian'])
        ]
        
        for i, (label, value, color) in enumerate(indicators):
            bar_y = y + i * bar_spacing
            
            # Label
            label_surface = self.fonts['tiny'].render(label, True, self.colors['text_secondary'])
            self.screen.blit(label_surface, (x + 10, bar_y - 2))
            
            # Background bar
            pygame.draw.rect(self.screen, self.colors['bg_tertiary'], 
                           (x + 80, bar_y, bar_width - 70, bar_height))
            
            # Value bar
            value_width = int((bar_width - 70) * value)
            pygame.draw.rect(self.screen, color, 
                           (x + 80, bar_y, value_width, bar_height))
            
            # Value text
            value_text = f"{value:.2f}"
            value_surface = self.fonts['tiny'].render(value_text, True, self.colors['text_primary'])
            self.screen.blit(value_surface, (x + width - 40, bar_y - 2))
    
    def _render_resource_deployment(self, region, x: int, y: int, width: int, height: int):
        """Render deployed resources as icons"""
        resources = region.resources_deployed
        icon_size = 16
        icon_spacing = 20
        
        current_x = x + 10
        
        # Peacekeepers
        if resources['peacekeepers'] > 0:
            pygame.draw.circle(self.screen, self.colors['peacekeepers'], 
                             (current_x + icon_size // 2, y + height // 2), icon_size // 2)
            count_text = self.fonts['tiny'].render(str(resources['peacekeepers']), 
                                                 True, self.colors['text_primary'])
            self.screen.blit(count_text, (current_x - 5, y + height // 2 + 8))
            current_x += icon_spacing
        
        # Aid
        if resources['aid'] > 0:
            pygame.draw.rect(self.screen, self.colors['humanitarian'], 
                           (current_x, y + height // 2 - icon_size // 2, icon_size, icon_size))
            count_text = self.fonts['tiny'].render(str(resources['aid']), 
                                                 True, self.colors['text_primary'])
            self.screen.blit(count_text, (current_x - 5, y + height // 2 + 8))
            current_x += icon_spacing
        
        # Monitors
        if resources['monitors'] > 0:
            pygame.draw.polygon(self.screen, self.colors['info'], [
                (current_x + icon_size // 2, y + height // 2 - icon_size // 2),
                (current_x + icon_size, y + height // 2 + icon_size // 2),
                (current_x, y + height // 2 + icon_size // 2)
            ])
            count_text = self.fonts['tiny'].render(str(resources['monitors']), 
                                                 True, self.colors['text_primary'])
            self.screen.blit(count_text, (current_x - 5, y + height // 2 + 8))
    
    def _render_active_conflicts(self, region, x: int, y: int):
        """Render active conflict indicators"""
        conflict_size = 8
        for i, conflict in enumerate(region.active_conflicts):
            conflict_y = y + i * (conflict_size + 2)
            color = self.colors['alert_red']
            
            # Add blinking effect
            if int(self.animation_time * 4) % 2:
                pygame.draw.circle(self.screen, color, (x, conflict_y), conflict_size)
    
    def _render_region_connections(self, environment):
        """Render connections between regions showing relationships"""
        # Simple connection lines for now
        # Could be enhanced with trade routes, conflict spillover, etc.
        pass
    
    def render_sidebar(self, environment):
        """Render information sidebar"""
        sidebar_x = self.width - 280
        sidebar_width = 270
        
        # Background
        pygame.draw.rect(self.screen, self.colors['bg_secondary'], 
                        (sidebar_x, 0, sidebar_width, self.height))
        pygame.draw.line(self.screen, self.colors['ui_border'], 
                        (sidebar_x, 0), (sidebar_x, self.height), 2)
        
        current_y = 20
        
        # Title
        title = self.fonts['large'].render("Crisis Monitor", True, self.colors['text_primary'])
        self.screen.blit(title, (sidebar_x + 10, current_y))
        current_y += 40
        
        # Global statistics
        total_risk = sum(environment._calculate_conflict_probability(region) 
                        for region in environment.regions)
        avg_risk = total_risk / len(environment.regions)
        
        stats = [
            ("Total Risk Level", f"{total_risk:.1f}"),
            ("Average Risk", f"{avg_risk:.2f}"),
            ("Active Crises", str(environment.crisis_count)),
            ("Budget Remaining", f"${environment.remaining_budget:.0f}"),
            ("Peacekeepers Available", str(environment.peacekeepers_available)),
            ("Episode Step", f"{environment.current_step}/{environment.time_horizon}"),
            ("Episode Reward", f"{environment.episode_reward:.1f}")
        ]
        
        for label, value in stats:
            # Label
            label_surface = self.fonts['small'].render(label, True, self.colors['text_secondary'])
            self.screen.blit(label_surface, (sidebar_x + 10, current_y))
            
            # Value
            value_surface = self.fonts['small'].render(value, True, self.colors['text_primary'])
            value_rect = value_surface.get_rect()
            value_rect.right = sidebar_x + sidebar_width - 10
            value_rect.y = current_y
            self.screen.blit(value_surface, value_rect)
            
            current_y += 25
        
        current_y += 20
        
        # Recent actions
        actions_title = self.fonts['medium'].render("Recent Actions", True, self.colors['text_primary'])
        self.screen.blit(actions_title, (sidebar_x + 10, current_y))
        current_y += 30
        
        for action in list(environment.action_history)[-5:]:
            action_text = f"T{action['step']}: {action['action_type'][:12]}"
            action_surface = self.fonts['tiny'].render(action_text, True, self.colors['text_secondary'])
            self.screen.blit(action_surface, (sidebar_x + 15, current_y))
            current_y += 15
    
    def render_bottom_panel(self, environment):
        """Render bottom information panel"""
        panel_height = 120
        panel_y = self.height - panel_height
        
        # Background
        pygame.draw.rect(self.screen, self.colors['bg_secondary'], 
                        (0, panel_y, self.width, panel_height))
        pygame.draw.line(self.screen, self.colors['ui_border'], 
                        (0, panel_y), (self.width, panel_y), 2)
        
        # Performance metrics
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        if dt > 0:
            fps = 1.0 / dt
            self.fps_history.append(fps)
            self.frame_times.append(dt * 1000)
        
        # FPS display
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            fps_text = f"FPS: {avg_fps:.1f}"
            fps_surface = self.fonts['small'].render(fps_text, True, self.colors['text_secondary'])
            self.screen.blit(fps_surface, (10, panel_y + 10))
        
        # Controls help
        help_text = [
            "Controls: [Space] Pause | [R] Reset | [D] Debug | [M] Metrics | [ESC] Quit",
            "Click regions to select | Scroll to zoom | Arrow keys to pan"
        ]
        
        for i, text in enumerate(help_text):
            help_surface = self.fonts['tiny'].render(text, True, self.colors['text_secondary'])
            self.screen.blit(help_surface, (10, panel_y + 40 + i * 15))
    
    def render_effects(self):
        """Render visual effects and particles"""
        # Update and render particle system
        dt = self.clock.get_time() / 1000.0
        self.particle_system.update(dt)
        self.particle_system.render(self.screen)
        
        # Update visual effects
        for effect in self.visual_effects[:]:
            effect.elapsed += dt
            if effect.elapsed >= effect.duration:
                self.visual_effects.remove(effect)
    
    def add_crisis_effect(self, x: float, y: float, severity: float = 1.0):
        """Add visual effect for crisis event"""
        self.particle_system.add_explosion(x, y, self.colors['explosion'], severity)
    
    def add_success_effect(self, x: float, y: float):
        """Add visual effect for successful action"""
        self.particle_system.add_explosion(x, y, self.colors['success'], 0.5)
    
    def handle_input(self, events):
        """Handle user input for interaction"""
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
                elif event.key == pygame.K_m:
                    self.show_metrics = not self.show_metrics
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_SPACE:
                    # Pause functionality (to be implemented)
                    pass
                elif event.key == pygame.K_r:
                    # Reset functionality (to be implemented)
                    pass
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check if clicking on a region
                    mouse_x, mouse_y = event.pos
                    self.selected_region = self._get_region_at_pos(mouse_x, mouse_y)
            
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom functionality
                self.zoom_level += event.y * 0.1
                self.zoom_level = max(0.5, min(3.0, self.zoom_level))
    
    def _get_region_at_pos(self, x: int, y: int) -> Optional[int]:
        """Get region ID at screen position"""
        for i in range(12):  # Number of regions
            rx, ry, rw, rh = self.calculate_region_bounds(i)
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return i
        return None
    
    def render_complete_frame(self, environment):
        """Render complete frame with all components"""
        # Update animation time
        self.animation_time += self.clock.get_time() / 1000.0
        
        # Clear screen
        self.screen.fill(self.colors['bg_primary'])
        
        # Render main components
        self.render_world_map(environment)
        self.render_sidebar(environment)
        self.render_bottom_panel(environment)
        self.render_effects()
        
        # Debug overlay
        if self.show_debug:
            self._render_debug_overlay(environment)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS
    
    def _render_debug_overlay(self, environment):
        """Render debug information overlay"""
        debug_info = [
            f"Zoom: {self.zoom_level:.2f}",
            f"Camera: ({self.camera_x}, {self.camera_y})",
            f"Selected: {self.selected_region}",
            f"Particles: {len(self.particle_system.particles)}",
            f"Effects: {len(self.visual_effects)}",
            f"Mode: {self.mode.value}"
        ]
        
        y_offset = 10
        for info in debug_info:
            surface = self.fonts['small'].render(info, True, self.colors['text_accent'])
            self.screen.blit(surface, (10, y_offset))
            y_offset += 20
    
    def cleanup(self):
        """Clean up rendering resources"""
        pygame.quit()
    
    def get_frame_as_array(self) -> np.ndarray:
        """Get current frame as numpy array for recording"""
        # Convert pygame surface to numpy array
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))  # Correct orientation
        return frame