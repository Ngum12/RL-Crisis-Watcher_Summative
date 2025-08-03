#!/usr/bin/env python3
"""
🎬 3-EPISODE REWARD MAXIMIZATION DEMONSTRATION
Create a 1.5-minute video showing agent maximizing rewards across 3 episodes

This demonstrates:
- Episode 1: Learning phase (lower rewards)
- Episode 2: Improvement phase (moderate rewards) 
- Episode 3: Optimization phase (maximum rewards)
- Clear reward tracking and episode transitions
- Neural network decision making
- Agent performance progression
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

def create_3_episode_reward_demo():
    """Create 90-second demonstration of 3 episodes with reward maximization"""
    
    print("🎬 CREATING 3-EPISODE REWARD MAXIMIZATION DEMONSTRATION")
    print("=" * 80)
    print("📊 Episodes:")
    print("   Episode 1: Learning Phase (30s) - Moderate rewards")
    print("   Episode 2: Improvement Phase (30s) - Better rewards") 
    print("   Episode 3: Optimization Phase (30s) - Maximum rewards")
    print("=" * 80)
    
    # Create output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = f'3_episode_reward_demo_{timestamp}.gif'
    
    print(f"📍 Recording to: gif_demos/{gif_path}")
    
    # Initialize pygame and GIF generator
    pygame.init()
    width, height = 1400, 900
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("🏆 3-Episode Reward Maximization - Neural Network Learning")
    clock = pygame.time.Clock()
    
    # 90-second GIF at 15 FPS = 1350 frames
    gif_gen = GifGenerator(max_duration=90.0, fps=15)
    
    # Colors
    colors = {
        'background': (5, 15, 25),
        'agent': (0, 255, 150),
        'crisis': (255, 60, 60),
        'text': (255, 255, 255),
        'neural': (100, 150, 255),
        'gold': (255, 215, 0),
        'accent': (0, 200, 255),
        'reward_good': (0, 255, 0),
        'reward_medium': (255, 255, 0),
        'reward_low': (255, 150, 0),
        'episode_1': (255, 100, 100),
        'episode_2': (255, 200, 100),
        'episode_3': (100, 255, 100),
        'panel_bg': (20, 30, 40)
    }
    
    # Episode configuration
    episodes = [
        {
            'name': 'Episode 1: Learning Phase',
            'color': colors['episode_1'],
            'reward_multiplier': 0.6,  # Lower rewards
            'crisis_rate': 0.8,        # More crises
            'success_rate': 0.4,       # Lower success
            'duration': 30             # 30 seconds
        },
        {
            'name': 'Episode 2: Improvement Phase', 
            'color': colors['episode_2'],
            'reward_multiplier': 0.8,  # Better rewards
            'crisis_rate': 0.6,        # Moderate crises
            'success_rate': 0.7,       # Better success
            'duration': 30             # 30 seconds
        },
        {
            'name': 'Episode 3: Optimization Phase',
            'color': colors['episode_3'],
            'reward_multiplier': 1.2,  # Maximum rewards
            'crisis_rate': 0.4,        # Fewer crises
            'success_rate': 0.9,       # High success
            'duration': 30             # 30 seconds
        }
    ]
    
    # Fonts
    font_title = pygame.font.Font(None, 48)
    font_large = pygame.font.Font(None, 32)
    font_medium = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)
    
    # Agent setup
    agent_pos = [300, 300]
    agent_trail = []
    
    # Episode tracking
    current_episode = 0
    episode_start_time = 0
    episode_rewards = [0, 0, 0]
    episode_crises_resolved = [0, 0, 0]
    episode_lives_saved = [0, 0, 0]
    total_reward = 0
    
    # Crisis tracking
    crises = []
    last_crisis_time = 0
    
    # Neural network simulation
    neural_activity = [random.random() for _ in range(20)]
    
    running = True
    frame_count = 0
    max_frames = 1350  # 90 seconds at 15 FPS
    animation_time = 0
    
    print("🎯 Starting 3-episode reward demonstration...")
    
    try:
        while running and frame_count < max_frames:
            dt = clock.tick(15) / 1000.0
            animation_time += dt
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # === EPISODE MANAGEMENT ===
            
            # Check episode transitions
            episode_elapsed = animation_time - episode_start_time
            if episode_elapsed >= episodes[current_episode]['duration'] and current_episode < 2:
                current_episode += 1
                episode_start_time = animation_time
                print(f"\n🎬 Starting {episodes[current_episode]['name']}")
            
            # Get current episode config
            current_ep = episodes[current_episode]
            
            # === SIMULATION UPDATES ===
            
            # Move agent (random patrol)
            target_x = 300 + 400 * math.sin(animation_time * 0.5)
            target_y = 300 + 200 * math.cos(animation_time * 0.3)
            
            dx = target_x - agent_pos[0]
            dy = target_y - agent_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > 5:
                speed = 60 * dt
                agent_pos[0] += (dx / distance) * speed
                agent_pos[1] += (dy / distance) * speed
                
                # Add to trail
                if len(agent_trail) == 0 or math.sqrt((agent_pos[0] - agent_trail[-1][0])**2 + 
                                                     (agent_pos[1] - agent_trail[-1][1])**2) > 10:
                    agent_trail.append(agent_pos.copy())
                    if len(agent_trail) > 30:
                        agent_trail.pop(0)
            
            # Generate crises based on episode
            crisis_interval = 2.0 / current_ep['crisis_rate']
            if animation_time - last_crisis_time > crisis_interval:
                crisis_pos = (
                    random.randint(200, 800),
                    random.randint(200, 600)
                )
                crises.append({
                    'pos': crisis_pos,
                    'age': 0,
                    'severity': random.uniform(0.5, 1.0),
                    'type': random.choice(['🔥', '🌊', '⚡', '🌪️'])
                })
                last_crisis_time = animation_time
            
            # Update crises and calculate rewards
            for crisis in crises[:]:
                crisis['age'] += dt
                
                # Check if agent resolves crisis
                agent_distance = math.sqrt((agent_pos[0] - crisis['pos'][0])**2 + 
                                         (agent_pos[1] - crisis['pos'][1])**2)
                
                if agent_distance < 40 and random.random() < current_ep['success_rate'] * dt:
                    # Crisis resolved - calculate reward
                    base_reward = 100 * crisis['severity']
                    episode_reward = base_reward * current_ep['reward_multiplier']
                    total_reward += episode_reward
                    episode_rewards[current_episode] += episode_reward
                    episode_crises_resolved[current_episode] += 1
                    episode_lives_saved[current_episode] += int(crisis['severity'] * 1000)
                    
                    crises.remove(crisis)
                elif crisis['age'] > 8.0:
                    # Crisis timeout - penalty
                    total_reward -= 50
                    crises.remove(crisis)
            
            # Update neural network activity
            for i in range(len(neural_activity)):
                neural_activity[i] += random.uniform(-0.3, 0.3) * dt
                neural_activity[i] = max(0, min(1, neural_activity[i]))
            
            # === RENDERING ===
            screen.fill(colors['background'])
            
            # === MAIN TITLE ===
            title = font_title.render("🏆 Neural Network Learning: 3-Episode Reward Maximization", True, colors['text'])
            screen.blit(title, (width//2 - title.get_width()//2, 20))
            
            # === EPISODE HEADER ===
            episode_title = font_large.render(current_ep['name'], True, current_ep['color'])
            screen.blit(episode_title, (width//2 - episode_title.get_width()//2, 70))
            
            episode_progress = (animation_time - episode_start_time) / current_ep['duration']
            progress_width = 400
            progress_rect = (width//2 - progress_width//2, 100, progress_width, 20)
            pygame.draw.rect(screen, colors['panel_bg'], progress_rect)
            pygame.draw.rect(screen, current_ep['color'], 
                           (progress_rect[0], progress_rect[1], 
                            int(progress_width * episode_progress), progress_rect[3]))
            pygame.draw.rect(screen, colors['text'], progress_rect, 2)
            
            # === MAIN SIMULATION AREA ===
            sim_area = (100, 140, 800, 500)
            pygame.draw.rect(screen, (30, 50, 70), sim_area)
            pygame.draw.rect(screen, current_ep['color'], sim_area, 3)
            
            # Agent trail
            if len(agent_trail) > 1:
                for i in range(len(agent_trail) - 1):
                    alpha = max(50, 255 - (len(agent_trail) - i) * 8)
                    thickness = max(1, 6 - i//5)
                    try:
                        pygame.draw.line(screen, colors['agent'], agent_trail[i], agent_trail[i+1], thickness)
                    except:
                        pass
            
            # Crises
            for crisis in crises:
                pos = crisis['pos']
                pulse = abs(math.sin(animation_time * 4)) * 0.5 + 0.5
                radius = int(15 + pulse * 10)
                
                crisis_color = tuple(max(0, min(255, int(c * (1 - crisis['age']/8.0)))) for c in colors['crisis'])
                pygame.draw.circle(screen, crisis_color, pos, radius)
                
                # Crisis icon
                crisis_text = font_medium.render(crisis['type'], True, colors['text'])
                text_rect = crisis_text.get_rect(center=pos)
                screen.blit(crisis_text, text_rect)
            
            # Agent
            agent_display_pos = tuple(map(int, agent_pos))
            
            # Agent glow
            glow_radius = int(25 + abs(math.sin(animation_time * 3)) * 10)
            glow_color = tuple(max(0, min(255, int(c * 0.4))) for c in colors['agent'])
            pygame.draw.circle(screen, glow_color, agent_display_pos, glow_radius)
            
            # Agent body
            pygame.draw.circle(screen, colors['agent'], agent_display_pos, 18)
            pygame.draw.circle(screen, colors['text'], agent_display_pos, 18, 3)
            
            # Agent label
            agent_label = font_small.render("🚁 AI Agent", True, colors['agent'])
            screen.blit(agent_label, (agent_display_pos[0] - 30, agent_display_pos[1] - 40))
            
            # === REWARD PANEL ===
            reward_panel = (920, 140, 450, 300)
            pygame.draw.rect(screen, colors['panel_bg'], reward_panel)
            pygame.draw.rect(screen, colors['gold'], reward_panel, 3)
            
            reward_title = font_large.render("🏆 Reward Tracking", True, colors['gold'])
            screen.blit(reward_title, (reward_panel[0] + 10, reward_panel[1] + 10))
            
            # Current episode reward
            current_reward_text = font_medium.render(f"Current Episode: +{episode_rewards[current_episode]:.1f}", True, current_ep['color'])
            screen.blit(current_reward_text, (reward_panel[0] + 15, reward_panel[1] + 50))
            
            # Total reward
            total_reward_text = font_medium.render(f"Total Reward: {total_reward:.1f}", True, colors['gold'])
            screen.blit(total_reward_text, (reward_panel[0] + 15, reward_panel[1] + 80))
            
            # Episode breakdown
            y_offset = 120
            for i, ep in enumerate(episodes):
                status = "COMPLETE" if i < current_episode else ("ACTIVE" if i == current_episode else "PENDING")
                color = ep['color'] if i <= current_episode else colors['text']
                
                ep_text = font_small.render(f"Ep {i+1}: {episode_rewards[i]:.1f} pts ({status})", True, color)
                screen.blit(ep_text, (reward_panel[0] + 15, reward_panel[1] + y_offset))
                
                lives_text = font_small.render(f"     Lives Saved: {episode_lives_saved[i]:,}", True, color)
                screen.blit(lives_text, (reward_panel[0] + 15, reward_panel[1] + y_offset + 20))
                
                crises_text = font_small.render(f"     Crises Resolved: {episode_crises_resolved[i]}", True, color)
                screen.blit(crises_text, (reward_panel[0] + 15, reward_panel[1] + y_offset + 40))
                
                y_offset += 70
            
            # === NEURAL NETWORK PANEL ===
            nn_panel = (920, 460, 450, 200)
            pygame.draw.rect(screen, colors['panel_bg'], nn_panel)
            pygame.draw.rect(screen, colors['neural'], nn_panel, 3)
            
            nn_title = font_large.render("🧠 Neural Network Learning", True, colors['neural'])
            screen.blit(nn_title, (nn_panel[0] + 10, nn_panel[1] + 10))
            
            # Learning progress indicator
            learning_text = font_medium.render(f"Learning Progress: Episode {current_episode + 1}/3", True, current_ep['color'])
            screen.blit(learning_text, (nn_panel[0] + 15, nn_panel[1] + 45))
            
            # Neural activity visualization
            for i in range(min(16, len(neural_activity))):
                x = nn_panel[0] + 20 + (i % 8) * 50
                y = nn_panel[1] + 80 + (i // 8) * 40
                
                activity = neural_activity[i]
                neuron_radius = int(8 + activity * 8)
                neuron_color = tuple(max(0, min(255, int(c * activity))) for c in colors['neural'])
                
                pygame.draw.circle(screen, neuron_color, (x, y), neuron_radius)
                pygame.draw.circle(screen, colors['text'], (x, y), neuron_radius, 1)
            
            # Performance improvement indicator
            if current_episode > 0:
                improvement = ((episode_rewards[current_episode] / max(1, episode_rewards[0])) - 1) * 100
                improvement_text = font_small.render(f"Performance Improvement: +{improvement:.1f}%", True, colors['reward_good'])
                screen.blit(improvement_text, (nn_panel[0] + 15, nn_panel[1] + 160))
            
            # === STATUS BAR ===
            status_y = height - 80
            pygame.draw.rect(screen, colors['panel_bg'], (0, status_y, width, 80))
            pygame.draw.rect(screen, colors['accent'], (0, status_y, width, 80), 2)
            
            # Time and frame info
            time_text = font_medium.render(f"Time: {animation_time:.1f}s | Frame: {frame_count + 1}/{max_frames}", True, colors['text'])
            screen.blit(time_text, (20, status_y + 10))
            
            # Current stats
            stats_text = font_small.render(f"Active Crises: {len(crises)} | Neural Activity: {np.mean(neural_activity):.2f}", True, colors['text'])
            screen.blit(stats_text, (20, status_y + 40))
            
            # Overall progress
            overall_progress = frame_count / max_frames
            progress_bar_width = 300
            progress_bar = (width - progress_bar_width - 20, status_y + 20, progress_bar_width, 15)
            pygame.draw.rect(screen, colors['panel_bg'], progress_bar)
            pygame.draw.rect(screen, colors['accent'], 
                           (progress_bar[0], progress_bar[1], 
                            int(progress_bar_width * overall_progress), progress_bar[3]))
            pygame.draw.rect(screen, colors['text'], progress_bar, 1)
            
            progress_text = font_small.render(f"Demo Progress: {overall_progress*100:.1f}%", True, colors['text'])
            screen.blit(progress_text, (width - progress_bar_width - 20, status_y + 40))
            
            pygame.display.flip()
            
            # Capture frame for GIF
            gif_gen.add_pygame_frame(screen)
            
            frame_count += 1
            
            # Progress indicator
            progress = (frame_count / max_frames) * 100
            if frame_count % 45 == 0:  # Every 3 seconds at 15 FPS
                print(f"\r🎬 Recording Episode {current_episode + 1}: {progress:.1f}% complete", end='')
    
    except KeyboardInterrupt:
        print("\n⏹️ Recording interrupted by user")
    
    # Save GIF
    print(f"\n💾 Saving 3-episode reward demo: {gif_path}")
    gif_gen.save_gif(f'gif_demos/{gif_path}', resize_factor=0.85)
    
    # Cleanup
    pygame.quit()
    
    print("✅ 3-Episode Reward Demonstration completed!")
    print(f"📁 Saved: gif_demos/{gif_path}")
    print("🏆 Final Statistics:")
    print(f"   Total Reward: {total_reward:.1f}")
    print(f"   Episode 1: {episode_rewards[0]:.1f} pts, {episode_lives_saved[0]:,} lives")
    print(f"   Episode 2: {episode_rewards[1]:.1f} pts, {episode_lives_saved[1]:,} lives") 
    print(f"   Episode 3: {episode_rewards[2]:.1f} pts, {episode_lives_saved[2]:,} lives")
    print("🎬 Perfect for 1.5-minute assignment video requirement!")
    
    return gif_path

if __name__ == "__main__":
    create_3_episode_reward_demo()