"""TRINETRA: physics + AI anomaly detection for Automatic Weather Station networks."""
from .engine import Trinetra, Panel
from .incidents import Analyzer
from .stream import StreamMonitor

__version__ = "1.0"
__all__ = ["Trinetra", "Panel", "Analyzer", "StreamMonitor"]
