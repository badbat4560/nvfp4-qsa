"""NVFP4 cache operations. CUDA/Triton imports are lazy; the codec also runs on CPU."""
from .oracle import encode, decode, global_scale_for
from .cache import PackedCache

__version__ = "0.1.0"
__all__ = ["encode", "decode", "global_scale_for", "PackedCache"]
