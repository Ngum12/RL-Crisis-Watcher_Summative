"""
Video and GIF Recording for Crisis Response AI

Advanced recording capabilities for creating demonstrations:
- High-quality video recording of training sessions
- GIF generation for reports and presentations
- Screen capture with audio support
- Automated highlight reel generation
- Performance-synchronized recordings
"""

import cv2
import numpy as np
import pygame
import threading
import time
from typing import List, Optional, Tuple, Callable
import imageio
from pathlib import Path
import os

class VideoRecorder:
    """Professional video recording for RL training demonstrations"""
    
    def __init__(self, output_path: str, fps: int = 30, quality: str = 'high'):
        self.output_path = output_path
        self.fps = fps
        self.quality = quality
        
        # Video settings
        self.quality_settings = {
            'low': {'bitrate': '1M', 'crf': 28},
            'medium': {'bitrate': '2M', 'crf': 23},
            'high': {'bitrate': '5M', 'crf': 18},
            'ultra': {'bitrate': '10M', 'crf': 15}
        }
        
        # Recording state
        self.is_recording = False
        self.frames = []
        self.audio_frames = []
        self.start_time = None
        self.frame_thread = None
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    def start_recording(self, width: int, height: int, enable_audio: bool = False):
        """Start video recording"""
        if self.is_recording:
            return
        
        self.width = width
        self.height = height
        self.enable_audio = enable_audio
        self.frames = []
        self.audio_frames = []
        self.start_time = time.time()
        self.is_recording = True
        
        print(f"Started recording: {self.output_path}")
    
    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """Add a frame to the recording"""
        if not self.is_recording:
            return
        
        # Ensure frame is correct format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Ensure frame is RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self.frames.append({
            'frame': frame.copy(),
            'timestamp': timestamp or time.time()
        })
    
    def add_pygame_frame(self, screen: pygame.Surface):
        """Add frame from pygame surface"""
        if not self.is_recording:
            return
        
        # Convert pygame surface to numpy array
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))  # Correct orientation
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self.add_frame(frame)
    
    def stop_recording(self):
        """Stop recording and save video"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if len(self.frames) == 0:
            print("No frames recorded")
            return
        
        print(f"Saving video with {len(self.frames)} frames...")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, 
                             (self.width, self.height))
        
        # Write frames
        for frame_data in self.frames:
            frame = frame_data['frame']
            # Resize if necessary
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            out.write(frame)
        
        out.release()
        print(f"Video saved: {self.output_path}")
        
        # Clear frames to free memory
        self.frames = []
    
    def create_highlight_reel(self, performance_data: List[float], 
                            highlight_threshold: float = 0.8,
                            max_duration: float = 180.0) -> str:
        """Create highlight reel of best moments"""
        if not self.frames or not performance_data:
            return None
        
        # Find highlight moments
        highlights = []
        for i, score in enumerate(performance_data):
            if score >= highlight_threshold and i < len(self.frames):
                highlights.append((i, score))
        
        if not highlights:
            return None
        
        # Sort by performance and take top moments
        highlights.sort(key=lambda x: x[1], reverse=True)
        
        # Create highlight video
        highlight_path = self.output_path.replace('.mp4', '_highlights.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(highlight_path, fourcc, self.fps, 
                             (self.width, self.height))
        
        total_frames = int(max_duration * self.fps)
        frames_per_highlight = total_frames // min(len(highlights), 10)
        
        for frame_idx, score in highlights[:10]:
            start_idx = max(0, frame_idx - frames_per_highlight // 2)
            end_idx = min(len(self.frames), start_idx + frames_per_highlight)
            
            for i in range(start_idx, end_idx):
                frame = self.frames[i]['frame']
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                out.write(frame)
        
        out.release()
        return highlight_path

class GifGenerator:
    """Create animated GIFs for documentation and presentations"""
    
    def __init__(self, max_duration: float = 10.0, fps: int = 10):
        self.max_duration = max_duration
        self.fps = fps
        self.frames = []
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to GIF"""
        # Convert to PIL-compatible format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Ensure RGB format
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]  # Remove alpha channel
        
        self.frames.append(frame)
        
        # Limit frames to max duration
        max_frames = int(self.max_duration * self.fps)
        if len(self.frames) > max_frames:
            self.frames = self.frames[-max_frames:]
    
    def add_pygame_frame(self, screen: pygame.Surface):
        """Add frame from pygame surface"""
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))  # Correct orientation
        self.add_frame(frame)
    
    def save_gif(self, output_path: str, optimize: bool = True, 
                 resize_factor: float = 0.5):
        """Save frames as animated GIF"""
        if not self.frames:
            print("No frames to save")
            return
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Resize frames if needed
        processed_frames = []
        for frame in self.frames:
            if resize_factor != 1.0:
                new_height = int(frame.shape[0] * resize_factor)
                new_width = int(frame.shape[1] * resize_factor)
                frame = cv2.resize(frame, (new_width, new_height))
            processed_frames.append(frame)
        
        # Save as GIF
        duration = 1.0 / self.fps
        imageio.mimsave(output_path, processed_frames, duration=duration, loop=0)
        
        print(f"GIF saved: {output_path} ({len(processed_frames)} frames)")
    
    def create_training_progress_gif(self, training_episodes: List[int], 
                                   output_path: str):
        """Create GIF showing training progress over episodes"""
        # This would integrate with the visualization system
        # to create a GIF showing how performance improves over time
        pass

class ScreenRecorder:
    """Record screen activity with optional audio"""
    
    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None):
        self.region = region  # (x, y, width, height)
        self.is_recording = False
        self.recorder_thread = None
        self.frames = []
    
    def start_recording(self, fps: int = 30):
        """Start screen recording"""
        if self.is_recording:
            return
        
        self.fps = fps
        self.is_recording = True
        self.frames = []
        
        # Start recording thread
        self.recorder_thread = threading.Thread(target=self._record_loop)
        self.recorder_thread.start()
    
    def _record_loop(self):
        """Main recording loop"""
        try:
            import pyautogui
            
            while self.is_recording:
                screenshot = pyautogui.screenshot(region=self.region)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                self.frames.append({
                    'frame': frame,
                    'timestamp': time.time()
                })
                
                time.sleep(1.0 / self.fps)
                
        except ImportError:
            print("pyautogui not available for screen recording")
            self.is_recording = False
    
    def stop_recording(self, output_path: str):
        """Stop recording and save video"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.recorder_thread:
            self.recorder_thread.join()
        
        if not self.frames:
            print("No frames recorded")
            return
        
        # Save video
        height, width = self.frames[0]['frame'].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        for frame_data in self.frames:
            out.write(frame_data['frame'])
        
        out.release()
        print(f"Screen recording saved: {output_path}")

class DemoRecorder:
    """Specialized recorder for creating RL demos and presentations"""
    
    def __init__(self, output_dir: str = 'demos/'):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.video_recorder = None
        self.gif_generator = GifGenerator()
        
    def start_episode_recording(self, algorithm: str, episode: int):
        """Start recording a specific episode"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{algorithm}_episode_{episode}_{timestamp}.mp4"
        output_path = os.path.join(self.output_dir, filename)
        
        self.video_recorder = VideoRecorder(output_path)
        return output_path
    
    def record_training_session(self, algorithm: str, total_episodes: int,
                              screen_size: Tuple[int, int]):
        """Record entire training session"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{algorithm}_training_{timestamp}.mp4"
        output_path = os.path.join(self.output_dir, filename)
        
        self.video_recorder = VideoRecorder(output_path, fps=30)
        self.video_recorder.start_recording(screen_size[0], screen_size[1])
        
        return output_path
    
    def add_performance_overlay(self, frame: np.ndarray, 
                              episode: int, reward: float, 
                              additional_info: dict = None) -> np.ndarray:
        """Add performance information overlay to frame"""
        overlay_frame = frame.copy()
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)
        thickness = 2
        
        # Episode number
        text = f"Episode: {episode}"
        cv2.putText(overlay_frame, text, (10, 30), font, font_scale, color, thickness)
        
        # Reward
        text = f"Reward: {reward:.1f}"
        cv2.putText(overlay_frame, text, (10, 60), font, font_scale, color, thickness)
        
        # Additional info
        if additional_info:
            y_offset = 90
            for key, value in additional_info.items():
                text = f"{key}: {value}"
                cv2.putText(overlay_frame, text, (10, y_offset), font, font_scale, color, thickness)
                y_offset += 30
        
        return overlay_frame
    
    def create_comparison_video(self, algorithm_videos: List[str], 
                              output_path: str):
        """Create side-by-side comparison video of multiple algorithms"""
        if len(algorithm_videos) < 2:
            return
        
        # Load videos
        caps = [cv2.VideoCapture(video) for video in algorithm_videos]
        
        # Get video properties
        fps = int(caps[0].get(cv2.CAP_PROP_FPS))
        frame_count = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)
        
        # Calculate output dimensions
        frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if len(algorithm_videos) == 2:
            output_width = frame_width * 2
            output_height = frame_height
        else:  # 4 algorithms in 2x2 grid
            output_width = frame_width * 2
            output_height = frame_height * 2
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        for _ in range(frame_count):
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))
            
            # Combine frames
            if len(frames) == 2:
                combined = np.hstack(frames)
            else:  # 4 frames in 2x2
                top_row = np.hstack(frames[:2])
                bottom_row = np.hstack(frames[2:4] if len(frames) >= 4 else 
                                     [frames[2], np.zeros_like(frames[0])])
                combined = np.vstack([top_row, bottom_row])
            
            out.write(combined)
        
        # Cleanup
        for cap in caps:
            cap.release()
        out.release()
        
        print(f"Comparison video saved: {output_path}")
    
    def generate_demo_package(self, algorithm_results: dict):
        """Generate complete demo package with videos, GIFs, and documentation"""
        package_dir = os.path.join(self.output_dir, 'demo_package')
        Path(package_dir).mkdir(parents=True, exist_ok=True)
        
        # Create individual algorithm demos
        for algorithm, data in algorithm_results.items():
            if 'best_episode_frames' in data:
                # Create GIF for best episode
                gif_path = os.path.join(package_dir, f'{algorithm}_best_episode.gif')
                gif_gen = GifGenerator(max_duration=15.0)
                for frame in data['best_episode_frames']:
                    gif_gen.add_frame(frame)
                gif_gen.save_gif(gif_path)
        
        # Create README with demo descriptions
        readme_path = os.path.join(package_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write("# Crisis Response AI - Demo Package\n\n")
            f.write("This package contains demonstration videos and GIFs showing the trained agents in action.\n\n")
            
            for algorithm in algorithm_results.keys():
                f.write(f"## {algorithm}\n")
                f.write(f"- Best episode demonstration: `{algorithm}_best_episode.gif`\n")
                f.write(f"- Training video: `{algorithm}_training.mp4` (if available)\n\n")
        
        return package_dir