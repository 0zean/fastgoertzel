"""Fast Goertzel Algorithm Implementation."""

from ._fastgoertzel_core import goertzel, goertzel_batch, goertzel_sliding_batch

__version__ = "1.0.2"
__all__ = ["goertzel", "goertzel_batch", "goertzel_sliding_batch"]
