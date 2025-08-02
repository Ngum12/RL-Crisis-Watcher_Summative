# Evaluation Package for Crisis Response AI
"""
Comprehensive evaluation and analysis tools for RL algorithms:

- Performance evaluation with multiple metrics
- Statistical significance testing
- Hyperparameter optimization
- Robustness testing
- Generalization analysis
- Comparative analysis tools
"""

from .performance_evaluator import PerformanceEvaluator, EvaluationMetrics
from .hyperparameter_optimizer import HyperparameterOptimizer, OptimizationResult
from .statistical_analysis import StatisticalAnalyzer, SignificanceTest
from .robustness_tester import RobustnessTester, RobustnessMetrics

__all__ = [
    'PerformanceEvaluator',
    'EvaluationMetrics',
    'HyperparameterOptimizer',
    'OptimizationResult',
    'StatisticalAnalyzer',
    'SignificanceTest',
    'RobustnessTester',
    'RobustnessMetrics'
]