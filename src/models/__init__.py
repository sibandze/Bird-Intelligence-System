# src/models/__init__.py
"""
Model definitions package for Bird-Intelligence-System.
Contains model architectures and related utilities.
"""

from .supervised_transformer import SupervisedTransformer
from .ssl.simclr import SimCLR

__all__ = ["SupervisedTransformer", "SimCLR"]
