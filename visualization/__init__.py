# Advanced Visualization Package for Crisis Response AI
"""
Sophisticated visualization components for the Crisis Response RL environment.
This package provides:
- Real-time 2D rendering with Pygame
- Advanced 3D visualization capabilities
- Interactive dashboard components
- Performance metrics visualization
- Training progress monitoring
- Agent behavior analysis tools

Components:
- real_time_renderer: Main rendering engine for environment visualization
- dashboard: Interactive control panel and metrics display
- charts: Performance charts and statistical visualizations
- 3d_viewer: Advanced 3D scene rendering (optional)
- recording: Video and GIF generation for demonstrations
"""

from .real_time_renderer import CrisisRenderer, RenderMode
from .dashboard import CrisisDashboard, MetricsPanel
from .charts import PerformanceCharts, TrainingVisualizer
from .recording import VideoRecorder, GifGenerator

__all__ = [
    'CrisisRenderer',
    'RenderMode',
    'CrisisDashboard', 
    'MetricsPanel',
    'PerformanceCharts',
    'TrainingVisualizer',
    'VideoRecorder',
    'GifGenerator'
]