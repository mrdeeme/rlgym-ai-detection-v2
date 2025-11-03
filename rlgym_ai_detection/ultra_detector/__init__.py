"""Ultra AI Detector — v2.3
Advanced heuristic+statistical LLM vs. Human detector with:
- Calibration and windowing
- Bootstrap confidence intervals
- POS entropy and semantic drift analysis
- Voice consistency detection
- CSV pipeline for batch processing
"""
from .detector import UltraAIDetector, DetectorConfig, DetectionReport
from .calibration import Calibrator
from .version import __version__

__all__ = ["UltraAIDetector", "DetectorConfig", "DetectionReport", "Calibrator", "__version__"]

