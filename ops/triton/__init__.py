"""
Triton fused kernels for NED model optimization.
"""

try:
    from .layernorm import rms_norm_fn
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    rms_norm_fn = None

__all__ = [
    "rms_norm_fn", 
    "HAS_TRITON"
]
