# Crisis Prediction & Response Environment Package
"""
Custom Gym environment for conflict prediction and crisis response simulation.
This package implements a sophisticated environment where an AI agent learns to:
- Monitor multiple regions for conflict indicators
- Predict potential crises before they escalate
- Deploy resources efficiently
- Issue timely alerts and recommendations
- Coordinate response efforts

The environment is designed to be challenging yet realistic, incorporating:
- Multi-dimensional state spaces with temporal dependencies
- Complex action spaces with resource constraints
- Dynamic reward structures that balance multiple objectives
- Realistic simulation of geopolitical factors
"""

from .crisis_env import CrisisResponseEnv
from .conflict_predictor import ConflictPredictorEnv
from .multi_region_env import MultiRegionCrisisEnv

__all__ = [
    'CrisisResponseEnv',
    'ConflictPredictorEnv', 
    'MultiRegionCrisisEnv'
]