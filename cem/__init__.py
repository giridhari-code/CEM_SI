"""
CEM: Cognitive Emergence Machine.
Main Package Initializer.
"""

from cem.core.model import CEM, ModelOutput
from cem.config import CEMConfig # Assuming you will have a config file

__version__ = "0.1.0"
__all__ = ["CEM", "ModelOutput", "CEMConfig"]