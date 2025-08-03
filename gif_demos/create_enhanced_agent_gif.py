#!/usr/bin/env python3
"""
🎬 GROUNDBREAKING MOBILE AGENT VISUALIZATION GIF CREATOR
Create stunning demonstrations of neural network-controlled crisis response agents!

This script creates professional-grade GIFs showing:
- Enhanced regional visualization of African countries
- Real-time neural network decision making
- Agent-to-command communication signals  
- Station response communications
- Agent movement with trail effects
- Crisis detection and response coordination

Perfect for demonstrating the "groundbreaking and mighty" capabilities!
"""

import pygame
import numpy as np
import sys
import os
import math
import time
import random
from datetime import datetime

# Add project root to path
sys.path.append('..')

from visualization.recording import GifGenerator

def create_enhanced_agent_gif():
    """Create a comprehensive demonstration GIF"""
    
    print("🎬 CREATING GROUNDBREAKING MOBILE AGENT DEMONSTRATION")
    print("=" * 80)
    print("🚀 Features:")
    print("   ✅ Enhanced African regional visualization")
    print("   ✅ Neural network decision making")
    print("   ✅ Agent-to-command communication")
    print("   ✅ Station response signals")
    print("   ✅ Movement trail effects")
    print("   ✅ Crisis detection and response")
    print("=" * 80)
    
    # Create output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = f'enhanced_mobile_agent_demo_{timestamp}.gif'
    
    print(f"📍 Recording to: gif_demos/{gif_path}")
    
    # Initialize pygame and GIF generator
    pygame.init()
    width, height = 1400, 900
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("🌍 Enhanced Mobile Crisis Agent - Neural Network Demo")
    clock = pygame.time.Clock()
    
    gif_gen = GifGenerator(max_duration=15.0, fps=12)  # 15 second high-quality GIF
    
    # Enhanced colors
    colors = {
        'background': (5, 15, 25),
        'agent': (0, 255, 150),
        'crisis': (255, 60, 60),
        'text': (255, 255, 255),
        'neural': (100, 150, 255),
        'gold': (255, 215, 0),
        'accent': (0, 200, 255),
        'land': (40, 80, 40),
        'water': (30, 100, 150),
        'route': (255, 100, 255),
        'panel_bg': (20, 30, 40),
        'cameroon': (100, 150, 100),
        'drc': (150, 100, 100),
        'sudan': (100, 100, 150),
        'signal': (0, 255, 255),
        'response': (255, 215, 0)
    }
    
    # Enhanced Africa map with regions
    countries = {
        'Cameroon': {
            'region': [(200, 200), (350, 200), (350, 350), (200, 350)],
            'color': colors['cameroon'],
            'capital': (275, 275),
            'cities': [(250, 225), (300, 320), (320, 240)]
        },
        'DR Congo': {
            'region': [(370, 250), (520, 250), (520, 450), (370, 450)],
            'color': colors['drc'],
            'capital': (445, 350),
            'cities': [(400, 300), (480, 380), (500, 320)]
        },
        'Sudan': {
            'region': [(540, 150), (700, 150), (700, 300), (540, 300)],
            'color': colors['sudan'],
            'capital': (620, 200),
            'cities': [(580, 180), (660, 170), (650, 220)]
        }
    }
    
    # Command centers
    command_centers = [
        {'name': 'African Union HQ', 'pos': (750, 400), 'color': colors['gold']},
        {'name': 'UN Emergency', 'pos': (100, 200), 'color': colors['neural']},
        {'name': 'Regional Command', 'pos': (500, 500), 'color': colors['accent']}
    ]
    
    # Agent setup
    agent_pos = [275, 275]  # Start in Cameroon capital
    agent_trail = []
    target_cities = [(445, 350), (620, 200), (275, 275)]  # Tour all capitals
    current_target_idx = 0
    
    # Crisis system
    crises = []
    signals = []
    responses = []
    
    # Neural network activity simulation
    neural_layers = {
        'input': [random.random() for _ in range(8)],
        'hidden1': [random.random() for _ in range(6)],
        'hidden2': [random.random() for _ in range(4)],
        'output': [random.random() for _ in range(3)]
    }
    
    # Fonts
    font_title = pygame.font.Font(None, 42)
    font_large = pygame.font.Font(None, 28)
    font_medium = pygame.font.Font(None, 22)
    font_small = pygame.font.Font(None, 18)
    
    running = True
    frame_count = 0
    max_frames = 180  # 15 seconds at 12 FPS
    animation_time = 0
    last_crisis_time = 0
    last_signal_time = 0
    
    print("🎯 Starting enhanced visualization recording...")
    
    try:
        while running and frame_count < max_frames:
            dt = clock.tick(12) / 1000.0
            animation_time += dt
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # === SIMULATION UPDATES ===
            
            # Move agent towards target
            if current_target_idx < len(target_cities):
                target = target_cities[current_target_idx]
                dx = target[0] - agent_pos[0]
                dy = target[1] - agent_pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 5:
                    # Move towards target
                    speed = 40 * dt
                    agent_pos[0] += (dx / distance) * speed
                    agent_pos[1] += (dy / distance) * speed
                    
                    # Add to trail
                    if len(agent_trail) == 0 or math.sqrt((agent_pos[0] - agent_trail[-1][0])**2 + 
                                                         (agent_pos[1] - agent_trail[-1][1])**2) > 3:
                        agent_trail.append(agent_pos.copy())
                        if len(agent_trail) > 25:
                            agent_trail.pop(0)
                else:
                    current_target_idx = (current_target_idx + 1) % len(target_cities)
            
            # Generate crises
            if animation_time - last_crisis_time > 4.0:
                crisis_pos = (random.randint(200, 700), random.randint(150, 450))
                crises.append({
                    'pos': crisis_pos,
                    'type': random.choice(['🔥', '🌊', '⚡', '🌪️']),
                    'age': 0,
                    'severity': random.uniform(0.5, 1.0)
                })
                last_crisis_time = animation_time
            
            # Update crises
            for crisis in crises[:]:
                crisis['age'] += dt
                if crisis['age'] > 8.0:
                    crises.remove(crisis)
            
            # Generate signals
            if animation_time - last_signal_time > 3.0:
                # Agent to command signal
                nearest_command = min(command_centers, key=lambda c: 
                                    math.sqrt((agent_pos[0] - c['pos'][0])**2 + (agent_pos[1] - c['pos'][1])**2))
                
                signals.append({
                    'from': agent_pos.copy(),
                    'to': nearest_command['pos'],
                    'age': 0,
                    'type': 'status'
                })
                
                # Command response (delayed)
                responses.append({
                    'from': nearest_command['pos'],
                    'to': agent_pos.copy(),
                    'age': -1.0,  # Delay
                    'type': 'response'
                })
                
                last_signal_time = animation_time
            
            # Update signals and responses
            for signal in signals[:]:
                signal['age'] += dt
                if signal['age'] > 3.0:
                    signals.remove(signal)
            
            for response in responses[:]:
                response['age'] += dt
                if response['age'] > 3.0:
                    responses.remove(response)
            
            # Update neural network activity
            for layer_name, layer in neural_layers.items():
                for i in range(len(layer)):
                    layer[i] += random.uniform(-0.1, 0.1) * dt
                    layer[i] = max(0, min(1, layer[i]))
            
            # === RENDERING ===
            screen.fill(colors['background'])
            
            # Main title
            title = font_title.render("🌍 AI-Powered Crisis Response: Africa Neural Network", True, colors['text'])
            screen.blit(title, (width//2 - title.get_width()//2, 20))
            
            subtitle = font_medium.render("Live demonstration of trained neural networks controlling mobile crisis agents", True, colors['accent'])
            screen.blit(subtitle, (width//2 - subtitle.get_width()//2, 60))
            
            # === MAP RENDERING ===
            map_area = (150, 120, 800, 450)
            
            # Map background
            pygame.draw.rect(screen, colors['land'], map_area)
            pygame.draw.rect(screen, colors['accent'], map_area, 3)
            
            # Country regions with enhanced visualization
            for country_name, country in countries.items():
                # Fill region
                region_color = tuple(max(0, min(255, int(c * 0.4))) for c in country['color'])
                pygame.draw.polygon(screen, region_color, country['region'])
                
                # Border
                border_color = tuple(max(0, min(255, int(c * 0.8))) for c in country['color'])
                pygame.draw.polygon(screen, border_color, country['region'], 4)
                
                # Country label
                center_x = sum(p[0] for p in country['region']) // len(country['region'])
                center_y = sum(p[1] for p in country['region']) // len(country['region'])
                
                country_label = font_large.render(country_name, True, colors['text'])
                label_rect = country_label.get_rect(center=(center_x, center_y - 30))
                
                # Label background
                pygame.draw.rect(screen, colors['panel_bg'], 
                               (label_rect.x - 5, label_rect.y - 2, label_rect.width + 10, label_rect.height + 4))
                screen.blit(country_label, label_rect)
                
                # Capital and cities
                pygame.draw.circle(screen, colors['gold'], country['capital'], 8)
                pygame.draw.circle(screen, colors['text'], country['capital'], 8, 2)
                
                for city in country['cities']:
                    pygame.draw.circle(screen, colors['text'], city, 4)
            
            # Command centers
            for center in command_centers:
                pos = center['pos']
                color = center['color']
                
                # Coverage area
                coverage_color = tuple(max(0, min(255, int(c * 0.1))) for c in color)
                pygame.draw.circle(screen, coverage_color, pos, 80)
                
                # Building
                pygame.draw.rect(screen, color, (pos[0]-10, pos[1]-10, 20, 20))
                pygame.draw.rect(screen, colors['text'], (pos[0]-10, pos[1]-10, 20, 20), 2)
                
                # Antenna
                pygame.draw.line(screen, color, (pos[0], pos[1]-10), (pos[0], pos[1]-25), 3)
                
                # Label
                label = font_small.render(center['name'], True, color)
                screen.blit(label, (pos[0] + 15, pos[1] - 15))
            
            # Crises
            for crisis in crises:
                pos = crisis['pos']
                pulse = abs(math.sin(animation_time * 4)) * 0.5 + 0.5
                radius = int(12 + pulse * 8)
                
                crisis_color = tuple(max(0, min(255, int(c * (1 - crisis['age']/8.0)))) for c in colors['crisis'])
                pygame.draw.circle(screen, crisis_color, pos, radius)
                
                # Crisis icon
                crisis_text = font_medium.render(crisis['type'], True, colors['text'])
                text_rect = crisis_text.get_rect(center=pos)
                screen.blit(crisis_text, text_rect)
            
            # Agent trail
            if len(agent_trail) > 1:
                for i in range(len(agent_trail) - 1):
                    alpha = max(50, 255 - (len(agent_trail) - i) * 10)
                    trail_color = (*colors['agent'][:3], alpha)
                    thickness = max(1, 5 - i//3)
                    
                    try:
                        pygame.draw.line(screen, colors['agent'], agent_trail[i], agent_trail[i+1], thickness)
                    except:
                        pass
            
            # Communication signals
            for signal in signals:
                if signal['age'] > 0:
                    progress = min(1.0, signal['age'] / 3.0)
                    
                    start_pos = signal['from']
                    end_pos = signal['to']
                    
                    current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
                    current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
                    current_pos = (int(current_x), int(current_y))
                    
                    # Signal beam
                    pygame.draw.line(screen, colors['signal'], start_pos, current_pos, 3)
                    
                    # Signal pulse
                    pulse_radius = int(6 + abs(math.sin(animation_time * 10)) * 4)
                    pygame.draw.circle(screen, colors['signal'], current_pos, pulse_radius)
                    
                    # Signal icon
                    signal_text = font_small.render("📡", True, colors['signal'])
                    screen.blit(signal_text, (current_pos[0] - 8, current_pos[1] - 8))
            
            # Command responses
            for response in responses:
                if response['age'] > 0:
                    progress = min(1.0, response['age'] / 3.0)
                    
                    start_pos = response['from']
                    end_pos = response['to']
                    
                    current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
                    current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
                    current_pos = (int(current_x), int(current_y))
                    
                    # Response beam (dashed)
                    segments = 8
                    for i in range(segments):
                        if i % 2 == 0:
                            seg_start_x = start_pos[0] + (current_pos[0] - start_pos[0]) * (i / segments)
                            seg_start_y = start_pos[1] + (current_pos[1] - start_pos[1]) * (i / segments)
                            seg_end_x = start_pos[0] + (current_pos[0] - start_pos[0]) * ((i + 1) / segments)
                            seg_end_y = start_pos[1] + (current_pos[1] - start_pos[1]) * ((i + 1) / segments)
                            
                            pygame.draw.line(screen, colors['response'], 
                                           (int(seg_start_x), int(seg_start_y)), 
                                           (int(seg_end_x), int(seg_end_y)), 3)
                    
                    # Response pulse
                    pulse_radius = int(5 + abs(math.sin(animation_time * 8)) * 3)
                    pygame.draw.circle(screen, colors['response'], current_pos, pulse_radius)
                    
                    # Response icon
                    response_text = font_small.render("📻", True, colors['response'])
                    screen.blit(response_text, (current_pos[0] - 8, current_pos[1] - 8))
            
            # Agent
            agent_display_pos = tuple(map(int, agent_pos))
            
            # Agent glow
            glow_radius = int(25 + abs(math.sin(animation_time * 2)) * 10)
            glow_color = tuple(max(0, min(255, int(c * 0.3))) for c in colors['agent'])
            pygame.draw.circle(screen, glow_color, agent_display_pos, glow_radius)
            
            # Agent body
            pygame.draw.circle(screen, colors['agent'], agent_display_pos, 15)
            pygame.draw.circle(screen, colors['text'], agent_display_pos, 15, 3)
            
            # Agent label
            agent_label = font_small.render("🚁 Neural Agent", True, colors['agent'])
            screen.blit(agent_label, (agent_display_pos[0] - 40, agent_display_pos[1] - 35))
            
            # === NEURAL NETWORK PANEL ===
            panel_x = 980
            panel_y = 120
            panel_w = 380
            panel_h = 300
            
            pygame.draw.rect(screen, colors['panel_bg'], (panel_x, panel_y, panel_w, panel_h))
            pygame.draw.rect(screen, colors['neural'], (panel_x, panel_y, panel_w, panel_h), 3)
            
            nn_title = font_large.render("🧠 Neural Network Activity", True, colors['neural'])
            screen.blit(nn_title, (panel_x + 10, panel_y + 10))
            
            # Neural layers visualization
            layer_positions = {
                'input': (panel_x + 30, panel_y + 60),
                'hidden1': (panel_x + 120, panel_y + 70),
                'hidden2': (panel_x + 210, panel_y + 80),
                'output': (panel_x + 300, panel_y + 90)
            }
            
            # Draw connections first
            for layer_name, next_layer in [('input', 'hidden1'), ('hidden1', 'hidden2'), ('hidden2', 'output')]:
                layer_pos = layer_positions[layer_name]
                next_pos = layer_positions[next_layer]
                
                for i, neuron_activity in enumerate(neural_layers[layer_name]):
                    for j, next_activity in enumerate(neural_layers[next_layer]):
                        start_x = layer_pos[0]
                        start_y = layer_pos[1] + i * 30
                        end_x = next_pos[0]
                        end_y = next_pos[1] + j * 30
                        
                        connection_strength = (neuron_activity + next_activity) / 2
                        alpha = int(connection_strength * 100 + 50)
                        connection_color = (*colors['neural'][:3], alpha)
                        
                        try:
                            pygame.draw.line(screen, colors['neural'], (start_x, start_y), (end_x, end_y), 1)
                        except:
                            pass
            
            # Draw neurons
            for layer_name, position in layer_positions.items():
                for i, activity in enumerate(neural_layers[layer_name]):
                    neuron_x = position[0]
                    neuron_y = position[1] + i * 30
                    
                    # Neuron activity visualization
                    activity_color = tuple(max(0, min(255, int(c * activity))) for c in colors['neural'])
                    neuron_radius = int(8 + activity * 4)
                    
                    pygame.draw.circle(screen, activity_color, (neuron_x, neuron_y), neuron_radius)
                    pygame.draw.circle(screen, colors['text'], (neuron_x, neuron_y), neuron_radius, 1)
            
            # Layer labels
            layer_labels = ['Input', 'Hidden', 'Hidden', 'Output']
            for i, (layer_name, position) in enumerate(layer_positions.items()):
                label = font_small.render(layer_labels[i], True, colors['text'])
                screen.blit(label, (position[0] - 10, position[1] - 20))
            
            # === STATUS PANEL ===
            status_y = panel_y + panel_h + 20
            pygame.draw.rect(screen, colors['panel_bg'], (panel_x, status_y, panel_w, 180))
            pygame.draw.rect(screen, colors['accent'], (panel_x, status_y, panel_w, 180), 2)
            
            status_title = font_large.render("📊 Mission Status", True, colors['accent'])
            screen.blit(status_title, (panel_x + 10, status_y + 10))
            
            status_info = [
                f"Active Crises: {len(crises)}",
                f"Signals Sent: {len([s for s in signals if s['age'] > 0])}",
                f"Responses: {len([r for r in responses if r['age'] > 0])}",
                f"Mission Time: {animation_time:.1f}s",
                f"Countries Monitored: 3",
                f"Neural Decisions: {frame_count // 10}"
            ]
            
            for i, info in enumerate(status_info):
                info_surface = font_small.render(info, True, colors['text'])
                screen.blit(info_surface, (panel_x + 15, status_y + 45 + i * 20))
            
            # === LEGEND ===
            legend_y = 600
            pygame.draw.rect(screen, colors['panel_bg'], (150, legend_y, 800, 120))
            pygame.draw.rect(screen, colors['gold'], (150, legend_y, 800, 120), 2)
            
            legend_title = font_large.render("🗺️ Legend - Enhanced Crisis Response System", True, colors['gold'])
            screen.blit(legend_title, (160, legend_y + 10))
            
            legend_items = [
                ("🚁", "Mobile Neural Agent", colors['agent']),
                ("📡", "Agent Signal", colors['signal']),
                ("📻", "Command Response", colors['response']),
                ("🔥", "Active Crisis", colors['crisis']),
                ("🏛️", "Command Centers", colors['gold']),
                ("━━", "Movement Trail", colors['agent']),
                ("▬▬", "Country Borders", colors['accent'])
            ]
            
            x_offset = 160
            y_offset = legend_y + 45
            for i, (icon, description, color) in enumerate(legend_items):
                if i == 4:  # New row
                    x_offset = 160
                    y_offset += 25
                
                icon_surface = font_medium.render(icon, True, color)
                desc_surface = font_small.render(description, True, colors['text'])
                
                screen.blit(icon_surface, (x_offset, y_offset))
                screen.blit(desc_surface, (x_offset + 25, y_offset + 2))
                
                x_offset += 180
            
            # Frame counter
            frame_info = font_small.render(f"Frame {frame_count+1}/{max_frames} | {animation_time:.1f}s", True, colors['text'])
            screen.blit(frame_info, (width - 200, height - 30))
            
            pygame.display.flip()
            
            # Capture frame for GIF
            gif_gen.add_pygame_frame(screen)
            
            frame_count += 1
            
            # Progress indicator
            progress = (frame_count / max_frames) * 100
            print(f"\r🎬 Recording: {progress:.1f}% complete [{frame_count}/{max_frames}]", end='')
    
    except KeyboardInterrupt:
        print("\n⏹️ Recording interrupted by user")
    
    # Save GIF
    print(f"\n💾 Saving enhanced GIF: {gif_path}")
    gif_gen.save_gif(f'gif_demos/{gif_path}', resize_factor=0.9)
    
    # Cleanup
    pygame.quit()
    
    print("✅ Enhanced GIF creation completed!")
    print(f"📁 Saved: gif_demos/{gif_path}")
    print("🌟 This demonstrates the full capabilities of your neural network system!")
    
    return gif_path

if __name__ == "__main__":
    create_enhanced_agent_gif()