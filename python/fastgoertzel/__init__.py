"""Fast Goertzel Algorithm Implementation."""

from ._fastgoertzel_core import goertzel, goertzel_batch

__version__ = "0.1.0"
__all__ = ["goertzel", "goertzel_batch"]
