"""NED Model Package - Advanced SSM Implementation.

This package provides a comprehensive implementation of the NED (Neural Efficient Decoder) model,
featuring multi-head selective state spaces, mixture of experts, and hybrid attention mechanisms.
"""

from .configuration_ned import NedConfig
from .modeling_ned import (
    NedModel,
    NedForCausalLM, 
    NedPreTrainedModel,
    NedCache,
    MultiHeadSelectiveScan,
    MixtureOfExperts,
    AdvancedRMSNorm,
    AdvancedHybridAttention
)

__version__ = "1.0.0"
__all__ = [
    "NedConfig",
    "NedModel", 
    "NedForCausalLM",
    "NedPreTrainedModel",
    "NedCache",
    "MultiHeadSelectiveScan",
    "MixtureOfExperts", 
    "AdvancedRMSNorm",
    "AdvancedHybridAttention"
]
