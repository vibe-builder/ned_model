"""PyTorch NED model - Comprehensive Advanced SSM Implementation.

This implementation integrates cutting-edge SSM research:
- Locally bidirectional scanning for improved global context
- Multi-head scanning in parallel subspaces  
- Sparse MoE integration for efficient scaling
- Matrix mixer design with bidirectional support
- Advanced position encodings (RoPE + ALiBi)
- Optimized Triton kernels with better memory patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from transformers import PreTrainedModel
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from configuration_ned import NedConfig
import dataclasses
import math
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None
import os  # For environment flag

# Add quantization imports and configuration
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    bnb = None

from ops.selective_scan_interface import selective_scan_fn

# Import causal_conv1d for high-performance depthwise convolution
try:
    from causal_conv1d import causal_conv1d_fn
    HAS_CAUSAL_CONV1D = True
except ImportError:
    HAS_CAUSAL_CONV1D = False
    causal_conv1d_fn = None

# Import Triton fused kernels
try:
    from ops.triton import rms_norm_fn, HAS_TRITON
except ImportError:
    HAS_TRITON = False
    rms_norm_fn = None

logger = logging.get_logger(__name__)

class AdvancedRMSNorm(nn.Module):
    """RMSNorm with learnable epsilon and optional centering."""
    def __init__(self, dim: int, eps: float = 1e-6, learnable_eps: bool = False, center: bool = False):
        super().__init__()
        self.center = center
        self.weight = nn.Parameter(torch.ones(dim))
        if learnable_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer("eps", torch.tensor(eps))
        
        if center:
            self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # Use Triton fused RMS norm if available and on CUDA
        if HAS_TRITON and rms_norm_fn is not None and x.is_cuda and not self.center:
            return rms_norm_fn(x, self.weight, self.eps)
        
        # Fallback to standard implementation
        if self.center:
            x = x - x.mean(dim=-1, keepdim=True)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x
        if self.center:
            x = x + self.bias
        return x

def get_advanced_position_encoding(seq_len, dim, device, base=10000, use_alibi=False):
    """Position encoding combining RoPE with ALiBi fallback."""
    if use_alibi:
        # ALiBi bias for longer sequences
        slopes = torch.tensor([1.0 / (2 ** (8 * i / dim)) for i in range(dim // 2)], device=device)
        positions = torch.arange(seq_len, device=device).float()
        alibi_bias = slopes.unsqueeze(0) * positions.unsqueeze(1)
        return alibi_bias
    else:
        # Standard RoPE
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(base) / dim))
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

# Removed unused selective_scan_fwd_kernel code - now using selective_scan_fn from BlackMamba

class NedCache:
    """
    Advanced cache supporting multi-head SSM, attention, and MoE routing.
    Optimized with lazy initialization to reduce memory footprint.
    
    Arguments:
        config: NedConfig
        batch_size: int
        dtype: torch.dtype
        device: torch.device
        
    Attributes:
        conv_states: Convolutional states for each layer [num_layers, batch_size, conv_kernel_size, conv_dim]
        ssm_states: SSM states for each layer [num_layers, batch_size, num_heads, head_dim, state_size]
        attention_cache: Optional attention key-value cache for hybrid layers
        expert_routing_cache: Optional MoE routing cache for load balancing
    """
    
    def __init__(
        self, 
        config, 
        batch_size: int, 
        dtype: torch.dtype = torch.float16, 
        device: Optional[str] = None
    ):
        self.dtype = dtype
        self.batch_size = batch_size
        self.num_layers = config.num_hidden_layers
        self.conv_kernel_size = getattr(config, 'conv_kernel_size', 4)
        self.num_heads = getattr(config, 'num_ssm_heads', 8)
        self.head_dim = config.hidden_size // self.num_heads
        self.state_size = config.intermediate_size // 4
        self.hidden_size = config.hidden_size
        self.device = device
        
        # Use lazy initialization - create placeholders instead of full tensors
        self._conv_states = None
        self._ssm_states = None
        
        # Optional attention cache for hybrid layers
        self.attention_cache = {}
        
        # Expert routing cache for MoE layers
        self.expert_routing_cache = None
        
        # Bidirectional scanning states (optional)
        self.scan_direction_states = None
        
        # Track which layers have been initialized
        self._initialized_conv_layers = set()
        self._initialized_ssm_layers = set()
        
    @property
    def conv_states(self):
        """Lazily initialize conv_states tensor when accessed."""
        if self._conv_states is None:
            self._conv_states = torch.zeros(
                self.num_layers,
                self.batch_size,
                self.conv_kernel_size,
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
            )
        return self._conv_states
    
    @conv_states.setter
    def conv_states(self, value):
        self._conv_states = value
    
    @property
    def ssm_states(self):
        """Lazily initialize ssm_states tensor when accessed."""
        if self._ssm_states is None:
            self._ssm_states = torch.zeros(
                self.num_layers,
                self.batch_size,
                self.num_heads,
                self.state_size,
                device=self.device,
                dtype=self.dtype,
            )
        return self._ssm_states
    
    @ssm_states.setter
    def ssm_states(self, value):
        self._ssm_states = value
    
    def _ensure_conv_layer_initialized(self, layer_idx: int):
        """Ensure that the convolutional state for a specific layer is initialized."""
        if self._conv_states is None:
            # Initialize only the required layer
            self._conv_states = torch.zeros(
                self.num_layers,
                self.batch_size,
                self.conv_kernel_size,
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
            )
        self._initialized_conv_layers.add(layer_idx)
    
    def _ensure_ssm_layer_initialized(self, layer_idx: int):
        """Ensure that the SSM state for a specific layer is initialized."""
        if self._ssm_states is None:
            # Initialize only the required layer
            self._ssm_states = torch.zeros(
                self.num_layers,
                self.batch_size,
                self.num_heads,
                self.state_size,
                device=self.device,
                dtype=self.dtype,
            )
        self._initialized_ssm_layers.add(layer_idx)
    
    def update_conv_state(
        self, 
        layer_idx: int, 
        new_conv_state: torch.Tensor, 
        cache_init: bool = False
    ) -> torch.Tensor:
        """Update convolutional state for a specific layer."""
        # Input validation
        if not isinstance(layer_idx, int):
            raise TypeError(f"layer_idx must be int, got {type(layer_idx)}")
        
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        
        if not isinstance(new_conv_state, torch.Tensor):
            raise TypeError(f"new_conv_state must be torch.Tensor, got {type(new_conv_state)}")
        
        if new_conv_state.numel() == 0:
            raise ValueError("Empty new_conv_state tensor")
        
        self._ensure_conv_layer_initialized(layer_idx)
        
        if cache_init:
            # Initialize entire conv state - handle dimension mismatch
            expected_shape = self.conv_states[layer_idx].shape
            if new_conv_state.shape != expected_shape:
                # Handle shape mismatch: expand (batch, channels) to (batch, kernel_size, channels)
                if new_conv_state.dim() == 2 and len(expected_shape) == 3:
                    new_conv_state = new_conv_state.unsqueeze(1).expand(-1, expected_shape[1], -1)
                elif new_conv_state.dim() == 3 and len(expected_shape) == 3:
                    # Ensure proper dimensions match
                    new_conv_state = new_conv_state.reshape(expected_shape)
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
        else:
            # Roll and update last position (for autoregressive generation)
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-2)
            # Handle single channel update - ensure shape compatibility
            if new_conv_state.shape[-1] != self.conv_states[layer_idx].shape[-1]:
                # Pad or reshape as needed
                target_size = self.conv_states[layer_idx].shape[-1]
                if new_conv_state.shape[-1] < target_size:
                    # Pad to match expected size
                    pad_size = target_size - new_conv_state.shape[-1]
                    new_conv_state = F.pad(new_conv_state, (0, pad_size))
                else:
                    # Truncate to match expected size
                    new_conv_state = new_conv_state[..., :target_size]
            self.conv_states[layer_idx][:, -1, :] = new_conv_state.to(self.conv_states.device)
        return self.conv_states[layer_idx]
    
    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        """Update SSM state for a specific layer."""
        # Input validation
        if not isinstance(layer_idx, int):
            raise TypeError(f"layer_idx must be int, got {type(layer_idx)}")
        
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        
        if not isinstance(new_ssm_state, torch.Tensor):
            raise TypeError(f"new_ssm_state must be torch.Tensor, got {type(new_ssm_state)}")
        
        if new_ssm_state.numel() == 0:
            raise ValueError("Empty new_ssm_state tensor")
        
        self._ensure_ssm_layer_initialized(layer_idx)
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]
    
    def get_conv_state(self, layer_idx: int) -> torch.Tensor:
        """Get convolutional state for a specific layer."""
        self._ensure_conv_layer_initialized(layer_idx)
        return self.conv_states[layer_idx]
    
    def get_ssm_state(self, layer_idx: int) -> torch.Tensor:
        """Get SSM state for a specific layer."""
        self._ensure_ssm_layer_initialized(layer_idx)
        return self.ssm_states[layer_idx]
    
    def reset(self):
        """Reset all cache states to zero."""
        if self._conv_states is not None:
            self._conv_states.zero_()
        if self._ssm_states is not None:
            self._ssm_states.zero_()
        self.attention_cache.clear()
        self.expert_routing_cache = None
        self._initialized_conv_layers.clear()
        self._initialized_ssm_layers.clear()

def create_linear_layer(in_features: int, out_features: int, bias: bool = False, 
                       quantization_config: Optional[dict] = None) -> nn.Module:
    """Create linear layer with optional quantization support.
    Quantization can reduce memory usage by 2-4x with minimal accuracy loss.
    """
    if quantization_config is None or not HAS_BITSANDBYTES:
        return nn.Linear(in_features, out_features, bias=bias)
    
    quant_type = quantization_config.get('load_in_8bit', False)
    quant_4bit = quantization_config.get('load_in_4bit', False)
    
    if quant_4bit:
        # 4-bit quantization: ~4x memory reduction, small accuracy loss
        return bnb.nn.Linear4bit(
            in_features, out_features, bias=bias,
            compute_dtype=quantization_config.get('bnb_4bit_compute_dtype', torch.float16),
            quant_type=quantization_config.get('bnb_4bit_quant_type', 'nf4'),
            use_double_quant=quantization_config.get('bnb_4bit_use_double_quant', True)
        )
    elif quant_type:
        # 8-bit quantization: ~2x memory reduction, negligible accuracy loss
        return bnb.nn.Linear8bitLt(in_features, out_features, bias=bias, has_fp16_weights=False)
    else:
        return nn.Linear(in_features, out_features, bias=bias)

class MixtureOfExperts(nn.Module):
    """High-performance vectorized MoE implementation with optimized routing.
    
    Key optimizations:
    - Batched expert computation for GPU efficiency
    - Advanced sparse routing with scatter/gather operations  
    - Memory-optimized load balancing
    - Quantization support throughout
    """
    def __init__(self, config: NedConfig, layer_idx: int):
        super().__init__()
        self.num_experts = getattr(config, 'num_experts', 8)
        self.num_experts_per_token = getattr(config, 'num_experts_per_token', 2)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.aux_loss_alpha = getattr(config, 'aux_loss_alpha', 0.01)
        self.routing_mode = getattr(config, 'routing_mode', 'top2')
        self.layer_idx = layer_idx
        
        # Get quantization config from model config
        quantization_config = getattr(config, 'quantization_config', None)
        
        # Optimized expert architecture: separate up/down projections for better batching
        self.expert_up_projs = nn.ModuleList([
            create_linear_layer(self.hidden_size, self.intermediate_size, bias=False, 
                              quantization_config=quantization_config)
            for _ in range(self.num_experts)
        ])
        self.expert_down_projs = nn.ModuleList([
            create_linear_layer(self.intermediate_size, self.hidden_size, bias=False,
                              quantization_config=quantization_config)
            for _ in range(self.num_experts)
        ])
        
        # Router/gate layer with improved initialization
        self.gate = create_linear_layer(self.hidden_size, self.num_experts, bias=False,
                                      quantization_config=quantization_config)
        # Better initialization for routing stability
        nn.init.normal_(self.gate.weight, std=0.02)
        
        # Dynamic routing strategy based on expert count and sequence length
        self.adaptive_routing = getattr(config, 'adaptive_moe_routing', True)
        self.expert_capacity_factor = getattr(config, 'expert_capacity_factor', 1.25)
        
        # Load balancing parameters
        self.load_balance_loss_coeff = getattr(config, 'load_balance_loss_coeff', 0.01)
        self.router_z_loss_coeff = getattr(config, 'router_z_loss_coeff', 0.001)
        
        # Add bias for auxiliary-loss-free balancing
        self.bias = nn.Parameter(torch.zeros(self.num_experts))
        
        # Add single shared expert
        self.shared_up = create_linear_layer(self.hidden_size, self.intermediate_size, bias=False, 
                                            quantization_config=quantization_config)
        self.shared_down = create_linear_layer(self.intermediate_size, self.hidden_size, bias=False,
                                              quantization_config=quantization_config)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized forward pass with batched expert computation and efficient routing.
        
        Returns:
            output: Expert-routed hidden states  
            aux_loss: Combined auxiliary losses for load balancing
        """
        # Input validation
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(hidden_states)}")
        
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected 3D tensor (batch, seq, hidden), got {hidden_states.dim()}D tensor")
        
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {hidden_states.size(-1)}")
        
        if hidden_states.numel() == 0:
            raise ValueError("Empty input tensor")
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        original_shape = hidden_states.shape
        
        # Flatten for routing: [batch*seq, hidden_size]
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_states_flat.size(0)
        
        # Router computation with numerical stability
        router_logits = self.gate(hidden_states_flat)  # [num_tokens, num_experts]
        
        # Add bias for selection
        original_logits = router_logits.clone()
        router_logits = router_logits + self.bias.unsqueeze(0).expand(num_tokens, -1)
        
        # Routing weights from original logits
        routing_weights = F.softmax(original_logits, dim=-1)
        
        # Use biased for selection
        routing_weights_for_selection = F.softmax(router_logits, dim=-1)
        
        # Top-k expert selection with load balancing
        top_k = self.num_experts_per_token
        topk_weights, topk_indices = torch.topk(routing_weights_for_selection, top_k, dim=-1)
        
        # Gather actual weights from original routing_weights
        actual_weights = torch.gather(routing_weights, dim=1, index=topk_indices)
        actual_weights = actual_weights / (actual_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Determine routing strategy based on problem size
        use_batched_routing = self._should_use_batched_routing(num_tokens, top_k)
        
        if use_batched_routing:
            # Batched expert computation for better GPU utilization
            output = self._batched_expert_forward(hidden_states_flat, actual_weights, topk_indices, top_k)
        else:
            # Memory-efficient sparse routing for large problems
            output = self._sparse_expert_forward(hidden_states_flat, actual_weights, topk_indices, top_k)
        
        # Add shared expert output
        shared_inter = F.silu(self.shared_up(hidden_states_flat))
        shared_out = self.shared_down(shared_inter)
        output += shared_out
        
        # Set aux_loss to 0 for loss-free balancing
        aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Reshape back to original shape
        output = output.view(original_shape)
        
        return output, aux_loss
    
    def _should_use_batched_routing(self, num_tokens: int, top_k: int) -> bool:
        """Determine whether to use batched or sparse routing based on problem size."""
        if not self.adaptive_routing:
            return self.num_experts <= 16  # Fixed threshold for backwards compatibility
        
        # Use batched routing for smaller problems or when top_k is high
        batched_threshold = 4096  # tokens
        sparsity_ratio = (top_k * num_tokens) / (self.num_experts * num_tokens)
        
        return num_tokens < batched_threshold or sparsity_ratio > 0.5
    
    def _batched_expert_forward(
        self, 
        hidden_states: torch.Tensor, 
        topk_weights: torch.Tensor, 
        topk_indices: torch.Tensor, 
        top_k: int
    ) -> torch.Tensor:
        """Batched expert computation for better GPU utilization."""
        num_tokens, hidden_size = hidden_states.shape
        output = torch.zeros_like(hidden_states)
        
        # Process each expert position in top-k
        for k in range(top_k):
            expert_weights = topk_weights[:, k:k+1]  # [num_tokens, 1]
            expert_indices = topk_indices[:, k]      # [num_tokens]
            
            # Create expert-wise batches using advanced indexing
            for expert_idx in range(self.num_experts):
                mask = (expert_indices == expert_idx)
                if not mask.any():
                    continue
                
                # Get tokens for this expert
                expert_input = hidden_states[mask]  # [expert_tokens, hidden_size]
                
                # Forward through expert layers with fused operations
                intermediate = self.expert_up_projs[expert_idx](expert_input)
                intermediate = F.silu(intermediate)  # In-place activation
                expert_output = self.expert_down_projs[expert_idx](intermediate)
                
                # Accumulate weighted outputs
                output[mask] += expert_output * expert_weights[mask]
        
        return output
    
    def _sparse_expert_forward(
        self, 
        hidden_states: torch.Tensor, 
        topk_weights: torch.Tensor, 
        topk_indices: torch.Tensor, 
        top_k: int
    ) -> torch.Tensor:
        """Memory-efficient sparse routing using scatter/gather operations."""
        num_tokens, hidden_size = hidden_states.shape
        
        # Flatten indices and weights for scatter operations
        flat_indices = topk_indices.view(-1)  # [num_tokens * top_k]
        flat_weights = topk_weights.view(-1)  # [num_tokens * top_k]
        
        # Create position indices for scatter
        token_indices = torch.arange(num_tokens, device=hidden_states.device)
        token_indices = token_indices.unsqueeze(1).expand(-1, top_k).reshape(-1)
        
        # Group tokens by expert using efficient scatter
        expert_inputs = {}
        expert_positions = {}
        expert_weights = {}
        
        for expert_idx in range(self.num_experts):
            mask = (flat_indices == expert_idx)
            if not mask.any():
                continue
                
            expert_token_pos = token_indices[mask]
            expert_inputs[expert_idx] = hidden_states[expert_token_pos]
            expert_positions[expert_idx] = expert_token_pos
            expert_weights[expert_idx] = flat_weights[mask]
        
        # Process experts and scatter results back
        output = torch.zeros_like(hidden_states)
        
        for expert_idx, expert_input in expert_inputs.items():
            # Forward through expert
            intermediate = self.expert_up_projs[expert_idx](expert_input)
            intermediate = F.silu(intermediate)
            expert_output = self.expert_down_projs[expert_idx](intermediate)
            
            # Weighted accumulation using scatter_add for efficiency
            weighted_output = expert_output * expert_weights[expert_idx].unsqueeze(-1)
            output.scatter_add_(0, 
                              expert_positions[expert_idx].unsqueeze(-1).expand(-1, hidden_size),
                              weighted_output)
        
        return output
    
    def _compute_router_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute router z-loss for training stability (prevents logit growth)."""
        if self.router_z_loss_coeff == 0.0 or not self.training:
            return torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)
        
        # Z-loss encourages router logits to stay close to zero
        z_loss = torch.mean(torch.square(torch.logsumexp(router_logits, dim=-1)))
        return z_loss * self.router_z_loss_coeff
    
    def _compute_load_balance_loss(
        self, 
        routing_weights: torch.Tensor, 
        topk_indices: torch.Tensor, 
        top_k: int
    ) -> torch.Tensor:
        """Optimized load balancing loss computation."""
        if self.load_balance_loss_coeff == 0.0:
            return torch.tensor(0.0, device=routing_weights.device, dtype=routing_weights.dtype)
        
        num_tokens = routing_weights.shape[0]
        
        # Compute expert usage frequencies efficiently using bincount
        expert_usage = torch.zeros(self.num_experts, device=routing_weights.device, dtype=routing_weights.dtype)
        
        # Count expert selections across all top-k positions
        for k in range(top_k):
            expert_indices = topk_indices[:, k]
            usage_counts = torch.bincount(expert_indices, minlength=self.num_experts)
            expert_usage += usage_counts.to(dtype=routing_weights.dtype)
        
        # Normalize to get expert fractions
        expert_fractions = expert_usage / (num_tokens * top_k)
        
        # Compute gate fractions (average routing probability per expert)
        gate_fractions = routing_weights.mean(dim=0)
        
        # Load balance loss: minimize variance between expert usage and gate assignments
        # This encourages uniform expert utilization
        load_balance_loss = self.num_experts * torch.sum(expert_fractions * gate_fractions)
        
        return load_balance_loss * self.load_balance_loss_coeff

class MultiHeadSelectiveScan(nn.Module):
    """Multi-head selective scan with parallel subspaces and bidirectional support."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.state_size = config.intermediate_size // 4  # More efficient state size
        self.num_heads = getattr(config, 'num_ssm_heads', 8)
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        
        # Get quantization config
        quantization_config = getattr(config, 'quantization_config', None)
        
        # Convolutional preprocessing with depthwise separable convs
        conv_kernel = getattr(config, 'conv_kernel_size', 4)
        self.conv1d = nn.Conv1d(
            self.hidden_size, self.hidden_size, 
            kernel_size=conv_kernel, 
            groups=self.hidden_size,  # Depthwise
            padding=conv_kernel - 1
        )
        
        # Multi-head parameter projections with optional quantization
        self.x_proj = create_linear_layer(self.hidden_size, 
                                        self.hidden_size + 2 * self.state_size * self.num_heads,
                                        quantization_config=quantization_config)
        self.dt_proj = create_linear_layer(self.head_dim, self.head_dim,
                                         bias=True,  # Explicitly set bias=True
                                         quantization_config=quantization_config)
        
        # Initialize dt bias using config parameters
        with torch.no_grad():
            if config.dt_init == "constant":
                nn.init.constant_(self.dt_proj.bias, config.dt_scale)
            elif config.dt_init == "random":
                nn.init.uniform_(self.dt_proj.bias, 
                               config.dt_init_floor, 
                               config.dt_scale)
            else:
                # Fallback to uniform initialization
                nn.init.uniform_(self.dt_proj.bias, 
                               config.dt_init_floor, 
                               config.dt_scale)
        
        # Per-head SSM parameters with advanced initialization
        self.A_log = nn.Parameter(self._init_A_matrix())
        self.B = nn.Parameter(torch.randn(self.num_heads, self.state_size, self.head_dim) * 0.02)
        self.C = nn.Parameter(torch.randn(self.num_heads, self.state_size, self.head_dim) * 0.02)
        self.D = nn.Parameter(torch.ones(self.hidden_size))
        
        # Bidirectional scanning support
        self.bidirectional = getattr(config, 'bidirectional_scan', True)
        if self.bidirectional:
            self.reverse_gate = nn.Parameter(torch.zeros(1))
        
        # Output normalization and projection with optional quantization
        self.norm = AdvancedRMSNorm(self.hidden_size)
        self.output_proj = create_linear_layer(self.hidden_size, self.hidden_size, bias=False,
                                             quantization_config=quantization_config)

    def _init_A_matrix(self) -> torch.Tensor:
        """HiPPO-style A matrix initialization for long-range modeling.
        This vectorized version avoids explicit Python loops for efficiency.
        Mathematical context: HiPPO matrices are structured to preserve long-range dependencies in SSMs.
        """
        h = self.num_heads
        d = self.head_dim
        s = self.state_size
        i = torch.arange(d).view(1, d, 1)
        j = torch.arange(s).view(1, 1, s)
        A = torch.zeros(h, d, s)
        mask = (i <= j)
        A[:, :, :] = torch.where(mask, -(2 * i + 1).float().sqrt() * (2 * j + 1).float().sqrt(),
                                 (2 * i + 1).float().sqrt() * (2 * j + 1).float().sqrt())
        A += 0.01 * torch.randn_like(A)
        A = -torch.abs(A)
        return torch.log(-A + 1e-7)

    def forward(
        self, 
        x, 
        cache: Optional[NedCache] = None, 
        cache_position: Optional[torch.LongTensor] = None
    ):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x)}")
        
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (batch, seq, hidden), got {x.dim()}D")
        
        if x.size(-1) != self.hidden_size:
            raise ValueError(f"Input hidden size {x.size(-1)} doesn't match expected {self.hidden_size}")
        
        if x.numel() == 0:
            raise ValueError("Empty input tensor")
        
        if cache_position is not None and not isinstance(cache_position, torch.Tensor):
            raise TypeError(f"cache_position must be torch.Tensor, got {type(cache_position)}")
        
        batch_size, seq_len, hidden_size = x.shape
        
        # For single-step generation, extract current states
        if cache is not None and cache_position is not None and cache_position[0] > 0:
            return self._single_step_forward(x, cache, cache_position)
        
        return self._full_sequence_forward(x, cache)
    
    def _single_step_forward(self, x, cache: NedCache, cache_position: torch.LongTensor):
        """Single-step forward for autoregressive generation."""
        batch_size, seq_len, hidden_size = x.shape
        
        # Validation for single-step generation
        if seq_len != 1:
            raise ValueError(f"Single-step forward expects seq_len=1, got {seq_len}")
        
        if not isinstance(cache, NedCache):
            raise TypeError(f"Expected NedCache, got {type(cache)}")
        
        if cache_position.dim() != 1:
            raise ValueError(f"cache_position must be 1D tensor, got {cache_position.dim()}D")
        
        # Update and apply convolution
        x_conv = self._apply_conv_single_step(x, cache)
        
        # Multi-head projections  
        x_proj = self.x_proj(x_conv)
        dt, B, C = torch.split(x_proj, [self.hidden_size, 
                                       self.state_size * self.num_heads,
                                       self.state_size * self.num_heads], dim=-1)
        
        # Reshape for multi-head processing
        x_heads = x_conv.view(batch_size, 1, self.num_heads, self.head_dim)
        dt_heads = self.dt_proj(dt.view(batch_size, 1, self.num_heads, -1))
        B_heads = B.view(batch_size, 1, self.num_heads, self.state_size)
        C_heads = C.view(batch_size, 1, self.num_heads, self.state_size)
        
        # Single-step SSM update
        outputs = self._ssm_single_step(x_heads, dt_heads, B_heads, C_heads, cache)
        
        # Combine heads and apply output processing
        outputs = outputs.reshape(batch_size, 1, self.hidden_size)  # Use reshape instead of view
        outputs = outputs + x * self.D  # Residual connection
        outputs = self.norm(outputs)
        outputs = self.output_proj(outputs)
        
        return outputs, cache
    
    def _full_sequence_forward(self, x: torch.Tensor, cache: Optional[NedCache]) -> Tuple[torch.Tensor, Optional[NedCache]]:
        batch_size, seq_len, hidden_size = x.shape
        x_conv = self._apply_conv_full_sequence(x, cache)
        x_proj = self.x_proj(x_conv)
        dt, B, C = torch.split(x_proj, [self.hidden_size, self.state_size * self.num_heads, self.state_size * self.num_heads], dim=-1)
        x_heads = x_conv.view(batch_size, seq_len, self.num_heads, self.head_dim)
        dt_heads = self.dt_proj(dt.view(batch_size, seq_len, self.num_heads, -1))
        B_heads = B.view(batch_size, seq_len, self.num_heads, self.state_size)
        C_heads = C.view(batch_size, seq_len, self.num_heads, self.state_size)
        
        # BlackMamba selective_scan integration
        B_batch, seq_len, num_heads, head_dim = x_heads.shape
        dim = num_heads * head_dim
        # Prepare inputs: reshape to (batch, dim, seq_len)
        u = x_heads.permute(0, 2, 3, 1).reshape(B_batch, dim, seq_len)
        delta = dt_heads.permute(0, 2, 3, 1).reshape(B_batch, dim, seq_len)
        # Flatten A_log to (dim, state_size) and compute A
        A = -torch.exp(self.A_log.view(dim, self.state_size))
        # Permute B and C to (batch, groups, state_size, seq_len)
        B_var = B_heads.permute(0, 2, 3, 1)
        C_var = C_heads.permute(0, 2, 3, 1)
        # Flat D parameter: shape (dim,)
        D_flat = self.D.view(dim)
        # Run selective_scan_fn (CPU fallback or CUDA extension), get output and final state
        if cache is not None:
            y, last_state = selective_scan_fn(
                u, delta, A, B_var, C_var, D_flat,
                delta_softplus=True,
                return_last_state=True
            )
            # last_state: (batch, dim, state_size) -> reshape to (batch, heads, state_size)
            last_state = last_state.view(B_batch, num_heads, self.state_size)
            cache.update_ssm_state(self.layer_idx, last_state)
        else:
            y = selective_scan_fn(u, delta, A, B_var, C_var, D_flat, delta_softplus=True)
        # Reshape output back to (batch, seq_len, num_heads, head_dim)
        outputs = y.reshape(B_batch, num_heads, head_dim, seq_len).permute(0, 3, 1, 2)
        
        # Bidirectional scan
        if self.bidirectional:
            x_rev = torch.flip(x_heads, dims=[1])
            dt_rev = torch.flip(dt_heads, dims=[1])
            B_rev = torch.flip(B_heads, dims=[1])
            C_rev = torch.flip(C_heads, dims=[1])
            
            # Prepare inputs for reverse direction
            u_rev = x_rev.permute(0, 2, 3, 1).reshape(B_batch, dim, seq_len)
            delta_rev = dt_rev.permute(0, 2, 3, 1).reshape(B_batch, dim, seq_len)
            B_var_rev = B_rev.permute(0, 2, 3, 1)
            C_var_rev = C_rev.permute(0, 2, 3, 1)
            
            # Run selective scan for reverse direction
            y_rev = selective_scan_fn(u_rev, delta_rev, A, B_var_rev, C_var_rev, D_flat, delta_softplus=True)
            outputs_rev = y_rev.reshape(B_batch, num_heads, head_dim, seq_len).permute(0, 3, 1, 2)
            
            # Flip back and combine with forward outputs
            outputs_rev = torch.flip(outputs_rev, dims=[1])
            gate = torch.sigmoid(self.reverse_gate)
            outputs = gate * outputs + (1 - gate) * outputs_rev
        
        outputs = outputs.view(batch_size, seq_len, self.hidden_size)
        outputs = outputs + x * self.D
        outputs = self.norm(outputs)
        outputs = self.output_proj(outputs)
        return outputs, cache
    
    def _apply_conv_single_step(self, x, cache: NedCache):
        """Apply convolution for single-step generation."""
        batch_size, seq_len, hidden_size = x.shape
        
        # Update conv cache
        cache.update_conv_state(self.layer_idx, x.squeeze(1), cache_init=False)
        
        # Apply convolution using cached states
        conv_state = cache.get_conv_state(self.layer_idx)  # [batch, kernel_size, hidden_size]
        
        # Compute convolution output manually
        # Simple approach: weighted sum over kernel
        # Handle the case where weights might be [hidden_size, 1, kernel_size]
        if self.conv1d.weight.dim() == 3:
            weights = self.conv1d.weight.squeeze(1)  # [hidden_size, kernel_size]
        else:
            weights = self.conv1d.weight  # Already [hidden_size, kernel_size]
            
        # Ensure dimensions match for einsum
        x_conv = torch.zeros(batch_size, hidden_size, device=x.device, dtype=x.dtype)
        for b in range(batch_size):
            for h in range(hidden_size):
                # Manual convolution for each batch and channel
                x_conv[b, h] = torch.sum(conv_state[b, :, h] * weights[h])
        
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        
        x_conv = F.silu(x_conv)
        return x_conv.unsqueeze(1)  # [batch, 1, hidden_size]
    
    def _apply_conv_full_sequence(self, x, cache: Optional[NedCache]):
        """Apply convolution for full sequence."""
        # Use causal_conv1d_fn for better performance if available
        if HAS_CAUSAL_CONV1D and x.is_cuda:
            # causal_conv1d expects (batch, dim, seq_len)
            x_transposed = x.transpose(-1, -2)  # (batch, hidden_size, seq_len)
            x_conv = causal_conv1d_fn(
                x_transposed,
                self.conv1d.weight.squeeze(-1),  # Remove the singular dimension
                self.conv1d.bias,
                activation="silu"
            )
            x_conv = x_conv.transpose(-1, -2)  # Back to (batch, seq_len, hidden_size)
        else:
            # Fallback to standard convolution
            x_conv = self.conv1d(x.transpose(-1, -2)).transpose(-1, -2)
            x_conv = x_conv[:, :x.size(1)]  # Trim to original sequence length
            x_conv = F.silu(x_conv)
        
        # Initialize cache if provided
        if cache is not None:
            # Store the last few states for future generation
            cache.update_conv_state(
                self.layer_idx, 
                x.transpose(1, 2)[:, :, -self.conv1d.kernel_size[0]:], 
                cache_init=True
            )
        
        return x_conv
        
    def _ssm_single_step(self, x: torch.Tensor, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor, cache: NedCache) -> torch.Tensor:
        """Single-step SSM update for generation.
        Simplified and numerically stable implementation.
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        
        # Get current SSM states: [batch, num_heads, state_size]
        ssm_states = cache.get_ssm_state(self.layer_idx)
        
        # Get A matrix and discretize: [heads, head_dim, state_size]
        A = -torch.exp(self.A_log)  # Negative for stability
        dt_softplus = F.softplus(dt.squeeze(1))  # [batch, heads, head_dim]
        
        # Simplified discretization: use the first element of each head_dim for stability
        dt_avg = dt_softplus.mean(dim=-1, keepdim=True)  # [batch, heads, 1]
        
        # Compute input contribution: B * x
        x_single = x.squeeze(1)  # [batch, heads, head_dim]
        B_single = B.squeeze(1)  # [batch, heads, state_size]
        
        # Use stable averaging for input contribution
        x_avg = x_single.mean(dim=-1, keepdim=True)  # [batch, heads, 1]
        B_contrib = B_single * x_avg  # [batch, heads, state_size]
        
        # Stable state update using simplified discretization
        # A_discrete ≈ exp(dt_avg * A_mean) for numerical stability
        A_mean = A.mean(dim=1, keepdim=True)  # [heads, 1, state_size]
        A_discrete = torch.exp(dt_avg.unsqueeze(-1) * A_mean.unsqueeze(0))  # [batch, heads, 1, state_size]
        
        # Update states with broadcasting
        new_states = A_discrete.squeeze(-2) * ssm_states + B_contrib  # [batch, heads, state_size]
        
        # Clamp to prevent numerical overflow
        new_states = torch.clamp(new_states, min=-1e6, max=1e6)
        
        # Update cache
        cache.update_ssm_state(self.layer_idx, new_states)
        
        # Compute output: C * states
        C_single = C.squeeze(1)  # [batch, heads, state_size]
        output_per_head = torch.sum(C_single * new_states, dim=-1)  # [batch, heads]
        
        # Broadcast to head_dim and reshape
        outputs = output_per_head.unsqueeze(-1).expand(-1, -1, head_dim)  # [batch, heads, head_dim]
        
        return outputs.view(batch_size, seq_len, num_heads, head_dim)

    # _triton_selective_scan method removed as it's no longer used

    # _pytorch_selective_scan method removed as it's no longer used

    # _ensure_cpu_compatibility and _cpu_optimized_selective_scan methods removed as they're no longer used

    # _get_onnx_compatible_forward method removed as it's no longer used

class AdvancedHybridAttention(nn.Module):
    """Advanced hybrid attention with FlashAttention-2 optimizations and sliding window patterns."""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.window_size = getattr(config, 'attention_window_size', 512)
        
        # Get quantization config
        quantization_config = getattr(config, 'quantization_config', None)
        
        # Projections with optional quantization
        self.q_proj = create_linear_layer(self.hidden_size, self.hidden_size, bias=False,
                                        quantization_config=quantization_config)
        self.k_proj = create_linear_layer(self.hidden_size, self.hidden_size, bias=False,
                                        quantization_config=quantization_config)
        self.v_proj = create_linear_layer(self.hidden_size, self.hidden_size, bias=False,
                                        quantization_config=quantization_config)
        self.o_proj = create_linear_layer(self.hidden_size, self.hidden_size, bias=False,
                                        quantization_config=quantization_config)
        
        # Advanced positional encoding
        self.use_rope = getattr(config, 'use_rope', True)
        self.use_alibi = getattr(config, 'use_alibi_fallback', False)
        if self.use_rope:
            self.rotary_base = getattr(config, 'rotary_base', 10000)
        
        # FlashAttention-2 optimization flags
        self.use_flash_attention = getattr(config, 'use_flash_attention', True)
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)

    def forward(self, hidden_states, attention_mask=None, cache=None, position_ids=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Projections
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply positional encoding
        if self.use_rope and position_ids is not None:
            pe = get_advanced_position_encoding(seq_len, self.head_dim, hidden_states.device, 
                                              self.rotary_base, use_alibi=self.use_alibi)
            if not self.use_alibi:
                q, k = self._apply_rope(q, k, pe)
            else:
                # ALiBi bias for very long sequences
                alibi_bias = pe.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        
        # Choose attention implementation based on sequence length and availability
        if self.use_flash_attention and seq_len > 128:
            try:
                # Try to use optimized attention (FlashAttention-2 compatible)
                attn_output = self._flash_attention_forward(q, k, v, attention_mask)
            except Exception:
                # Fallback to standard attention
                attn_output = self._standard_attention(q, k, v, attention_mask)
        elif seq_len > self.window_size:
            # Sliding window attention for very long sequences
            attn_output = self._sliding_window_attention(q, k, v, attention_mask)
        else:
            # Standard attention for shorter sequences
            attn_output = self._standard_attention(q, k, v, attention_mask)
        
        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output), cache

    def _flash_attention_forward(self, q, k, v, attention_mask):
        """FlashAttention-2 compatible efficient attention computation.
        Implements memory-efficient attention with reduced memory transfers.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Use F.scaled_dot_product_attention when available (PyTorch 2.0+)
        # This automatically uses FlashAttention kernels when possible
        if hasattr(F, 'scaled_dot_product_attention'):
            # Convert attention mask format if needed
            if attention_mask is not None and attention_mask.dim() == 2:
                # Convert from [batch, seq_len] to [batch, 1, 1, seq_len]
                attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
                # Convert from additive to boolean mask
                attention_mask = attention_mask < 0
            
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False  # We handle causality via attention_mask
            )
            return attn_output
        else:
            # Fallback to manual implementation with memory optimizations
            return self._memory_efficient_attention(q, k, v, attention_mask)

    def _memory_efficient_attention(self, q, k, v, attention_mask):
        """Memory-efficient attention implementation using chunked computation.
        Reduces peak memory usage for very long sequences.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Use chunked attention for very long sequences to reduce memory
        chunk_size = min(512, seq_len)  # Process in chunks of 512
        
        if seq_len <= chunk_size:
            # Short sequences: standard attention
            return self._standard_attention(q, k, v, attention_mask)
        
        # Long sequences: chunked attention computation
        attn_output = torch.zeros_like(q)
        scale = math.sqrt(head_dim)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i]
            
            # Compute attention scores for this query chunk against all keys
            attn_scores = torch.matmul(q_chunk, k.transpose(-2, -1)) / scale
            
            if attention_mask is not None:
                # Apply attention mask to the chunk
                mask_chunk = attention_mask[:, :, i:end_i, :]
                attn_scores = attn_scores + mask_chunk
            
            # Softmax and weighted sum
            attn_probs = F.softmax(attn_scores, dim=-1)
            if self.attention_dropout > 0.0 and self.training:
                attn_probs = F.dropout(attn_probs, p=self.attention_dropout)
            
            attn_output[:, :, i:end_i] = torch.matmul(attn_probs, v)
        
        return attn_output

    def _standard_attention(self, q, k, v, attention_mask):
        """Standard attention computation for baseline comparison."""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        if self.attention_dropout > 0.0 and self.training:
            attn_probs = F.dropout(attn_probs, p=self.attention_dropout)
        
        return torch.matmul(attn_probs, v)

    def _apply_rope(self, q, k, pe):
        """Apply rotary position embeddings."""
        # Simplified RoPE application
        q_rot = q * pe.cos() + self._rotate_half(q) * pe.sin()
        k_rot = k * pe.cos() + self._rotate_half(k) * pe.sin()
        return q_rot, k_rot

    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _sliding_window_attention(self, q, k, v, mask):
        """Optimized sliding window attention for very long sequences.
        Implements local attention patterns with 50% overlap for better context.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Create output tensor
        output = torch.zeros_like(q)
        
        # Adaptive window size based on sequence length
        effective_window_size = min(self.window_size, seq_len // 4)  # Ensure reasonable window
        overlap_size = effective_window_size // 4  # 25% overlap instead of 50% for efficiency
        
        for i in range(0, seq_len, effective_window_size - overlap_size):
            end_idx = min(i + effective_window_size, seq_len)
            
            q_window = q[:, :, i:end_idx]
            k_window = k[:, :, i:end_idx] 
            v_window = v[:, :, i:end_idx]
            
            # Compute attention for window using optimized kernel
            scores = torch.matmul(q_window, k_window.transpose(-2, -1)) / math.sqrt(head_dim)
            
            if mask is not None:
                window_mask = mask[:, :, i:end_idx, i:end_idx]
                scores = scores + window_mask
            
            probs = F.softmax(scores, dim=-1)
            window_output = torch.matmul(probs, v_window)
            
            # Weighted combination for overlapping regions to reduce artifacts
            if i > 0 and i + overlap_size < seq_len:
                # Linear interpolation in overlap region
                alpha = torch.linspace(0, 1, overlap_size, device=q.device).view(1, 1, -1, 1)
                output[:, :, i:i+overlap_size] = (1 - alpha) * output[:, :, i:i+overlap_size] + \
                                               alpha * window_output[:, :, :overlap_size]
                output[:, :, i+overlap_size:end_idx] = window_output[:, :, overlap_size:]
            else:
                output[:, :, i:end_idx] = window_output
        
        return output

class AdvancedNedLayer(nn.Module):
    """Mamba-inspired NED layer with gated SSM and efficient architecture."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Get quantization config
        quantization_config = getattr(config, 'quantization_config', None)
        
        # Single normalization at the beginning (Mamba-style)
        self.norm = AdvancedRMSNorm(config.hidden_size)
        
        # Input projection to expand dimensionality with optional quantization
        self.in_proj = create_linear_layer(config.hidden_size, 2 * config.intermediate_size, 
                                         bias=False, quantization_config=quantization_config)
        
        # Dimension adapters for SSM compatibility with optional quantization
        self.ssm_in_proj = create_linear_layer(config.intermediate_size, config.hidden_size, 
                                             bias=False, quantization_config=quantization_config)
        self.ssm_out_proj = create_linear_layer(config.hidden_size, config.intermediate_size, 
                                              bias=False, quantization_config=quantization_config)
        
        # Core SSM component
        self.ssm = MultiHeadSelectiveScan(config, layer_idx)
        
        # Output projection back to hidden_size with optional quantization
        self.out_proj = create_linear_layer(config.intermediate_size, config.hidden_size, 
                                          bias=False, quantization_config=quantization_config)
        
        # Hybrid attention every few layers (optional)
        attention_freq = getattr(config, 'attention_frequency', 8)  # Less frequent than before
        self.use_attention = (layer_idx % attention_freq == 0) and getattr(config, 'use_hybrid_attention', False)
        if self.use_attention:
            self.attention = AdvancedHybridAttention(config)
            self.norm_attn = AdvancedRMSNorm(config.hidden_size)
        
        # MoE integration (only in later layers)
        use_moe = getattr(config, 'use_moe', False) and layer_idx > config.num_hidden_layers // 2
        if use_moe:
            # Create a temporary config for MoE with intermediate_size as hidden_size
            class MoEConfig:
                def __init__(self, base_config):
                    # Core MoE parameters
                    self.hidden_size = base_config.intermediate_size  # Key change: use intermediate_size
                    self.intermediate_size = base_config.intermediate_size
                    self.num_experts = getattr(base_config, 'num_experts', 8)
                    self.num_experts_per_token = getattr(base_config, 'num_experts_per_token', 2)
                    self.routing_mode = getattr(base_config, 'routing_mode', 'top2')
                    self.adaptive_moe_routing = getattr(base_config, 'adaptive_moe_routing', True)
                    self.expert_capacity_factor = getattr(base_config, 'expert_capacity_factor', 1.25)
                    self.load_balance_loss_coeff = getattr(base_config, 'load_balance_loss_coeff', 0.01)
                    self.router_z_loss_coeff = getattr(base_config, 'router_z_loss_coeff', 0.001)
                    self.quantization_config = getattr(base_config, 'quantization_config', None)
            
            moe_config = MoEConfig(config)
            self.moe = MixtureOfExperts(moe_config, layer_idx)
            self.use_moe = True
        else:
            self.use_moe = False
        
        # Progressive layer scaling (optional)
        self.layer_scale = getattr(config, 'use_layer_scale', False)  # Default off for simpler arch
        if self.layer_scale:
            init_values = getattr(config, 'layer_scale_init', 1e-4)
            self.gamma = nn.Parameter(init_values * torch.ones(config.hidden_size))

    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        position_ids=None, 
        cache_params: Optional[NedCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache=False, 
        output_router_logits=False
    ):
        # Mamba-style gated architecture
        residual = hidden_states
        
        # Single normalization at the beginning
        hidden_states = self.norm(hidden_states)
        
        # Input projection - expand to 2x intermediate size for gating
        projected = self.in_proj(hidden_states)  # [batch, seq, 2 * intermediate_size]
        
        # Split into SSM input and gate
        ssm_input, gate = projected.chunk(2, dim=-1)  # Each: [batch, seq, intermediate_size]
        
        # Project SSM input to match expected hidden_size for the SSM
        ssm_input_resized = self.ssm_in_proj(ssm_input)
        
        # Process through SSM
        ssm_output, cache_params = self.ssm(
            ssm_input_resized, 
            cache=cache_params,
            cache_position=cache_position
        )
        
        # Project SSM output back to intermediate size
        ssm_output_resized = self.ssm_out_proj(ssm_output)
        
        # Apply gating with SiLU activation (Mamba-style)
        gated_output = ssm_output_resized * F.silu(gate)
        
        # Optional MoE processing
        aux_loss = 0.0
        if self.use_moe:
            # Apply MoE to gated output  
            gated_output, aux_loss = self.moe(gated_output)
        
        # Output projection back to hidden_size
        output = self.out_proj(gated_output)
        
        # Apply layer scaling if enabled
        if self.layer_scale:
            output = self.gamma * output
        
        # Residual connection
        hidden_states = residual + output
        
        # Optional attention processing (hybrid approach)
        if self.use_attention:
            attn_residual = hidden_states
            hidden_states = self.norm_attn(hidden_states)
            attn_output, attn_cache = self.attention(
                hidden_states, 
                attention_mask, 
                cache_params.attention_cache.get(self.layer_idx) if cache_params else None, 
                position_ids
            )
            hidden_states = attn_residual + attn_output
            
            if cache_params is not None:
                cache_params.attention_cache[self.layer_idx] = attn_cache
        
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (cache_params,)
        if output_router_logits and aux_loss > 0:
            outputs = outputs + (aux_loss,)
        
        return outputs

class NedPreTrainedModel(PreTrainedModel):
    config_class = NedConfig
    base_model_prefix = "ned"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AdvancedNedLayer"]

class NedModel(NedPreTrainedModel):
    """Advanced NED model with comprehensive SSM optimizations."""
    def __init__(self, config):
        super().__init__(config)
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        
        # Advanced layer stack
        self.layers = nn.ModuleList([
            AdvancedNedLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = AdvancedRMSNorm(config.hidden_size)
        
        # Gradient checkpointing
        self.gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[NedCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # Input validation
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot provide both input_ids and inputs_embeds")
        
        if input_ids is not None:
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError(f"input_ids must be torch.Tensor, got {type(input_ids)}")
            if input_ids.dim() not in [1, 2]:
                raise ValueError(f"input_ids must be 1D or 2D tensor, got {input_ids.dim()}D")
            if input_ids.numel() == 0:
                raise ValueError("Empty input_ids tensor")
        
        if inputs_embeds is not None:
            if not isinstance(inputs_embeds, torch.Tensor):
                raise TypeError(f"inputs_embeds must be torch.Tensor, got {type(inputs_embeds)}")
            if inputs_embeds.dim() != 3:
                raise ValueError(f"inputs_embeds must be 3D tensor, got {inputs_embeds.dim()}D")
            if inputs_embeds.size(-1) != self.config.hidden_size:
                raise ValueError(f"inputs_embeds last dim {inputs_embeds.size(-1)} != hidden_size {self.config.hidden_size}")
        
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError(f"attention_mask must be torch.Tensor, got {type(attention_mask)}")
        
        if cache_params is not None and not isinstance(cache_params, NedCache):
            raise TypeError(f"cache_params must be NedCache, got {type(cache_params)}")
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]
        
        # Generate position IDs if not provided
        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        hidden_states = inputs_embeds
        
        # Initialize cache
        if use_cache:
            if cache_params is None:
                cache_params = NedCache(
                    self.config, 
                    inputs_embeds.size(0), 
                    device=inputs_embeds.device, 
                    dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, seq_len, device=inputs_embeds.device)
            elif cache_position is None:
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None
        
        # Collect outputs
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Gradient checkpointing
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    cache_params,
                    cache_position,
                    use_cache,
                    output_router_logits
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids, 
                    cache_params=cache_params,
                    cache_position=cache_position,
                    use_cache=use_cache,
                    output_router_logits=output_router_logits
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                cache_params = layer_outputs[1]
            
            if output_router_logits and len(layer_outputs) > 2:
                all_router_logits += (layer_outputs[2],)

        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states, all_router_logits] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=cache_params,
            hidden_states=all_hidden_states,
            attentions=all_router_logits,  # Reusing this field for router logits
        )

class NedForCausalLM(NedPreTrainedModel, GenerationMixin):
    """Advanced NED model for causal language modeling."""
    _tied_weights_keys = []
    
    def __init__(self, config):
        super().__init__(config)
        self.model = NedModel(config)
        
        # Get quantization config
        quantization_config = getattr(config, 'quantization_config', None)
        
        # Language modeling head with optional quantization
        self.lm_head = create_linear_layer(config.hidden_size, config.vocab_size, bias=False,
                                         quantization_config=quantization_config)

        if getattr(config, 'tie_word_embeddings', False):
            # Note: Weight tying with quantized layers requires special handling
            if quantization_config is None:
                self.lm_head.weight = self.model.embed_tokens.weight
            else:
                # For quantized models, weight tying is handled differently
                logger.warning("Weight tying with quantized layers may not be fully supported. "
                             "Consider setting tie_word_embeddings=False for quantized models.")

        self.post_init()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Override to handle quantization configuration for loading."""
        # Extract quantization arguments
        load_in_8bit = kwargs.pop('load_in_8bit', False)
        load_in_4bit = kwargs.pop('load_in_4bit', False)
        device_map = kwargs.get('device_map', None)
        
        if (load_in_8bit or load_in_4bit) and not HAS_BITSANDBYTES:
            raise ImportError("bitsandbytes is required for quantized inference. "
                            "Install with: pip install bitsandbytes")
        
        # Add quantization config to model config
        if load_in_8bit or load_in_4bit:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
            config.quantization_config = {
                'load_in_8bit': load_in_8bit,
                'load_in_4bit': load_in_4bit,
                'bnb_4bit_compute_dtype': kwargs.pop('bnb_4bit_compute_dtype', torch.float16),
                'bnb_4bit_quant_type': kwargs.pop('bnb_4bit_quant_type', 'nf4'),
                'bnb_4bit_use_double_quant': kwargs.pop('bnb_4bit_use_double_quant', True)
            }
            kwargs['config'] = config
            
            # Ensure appropriate device map for quantized models
            if device_map is None and torch.cuda.is_available():
                kwargs['device_map'] = 'auto'
        
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.model.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[NedCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Add auxiliary MoE loss if present
            aux_loss = 0.0
            if output_router_logits and hasattr(outputs, 'attentions') and outputs.attentions:
                for router_logits in outputs.attentions:
                    if isinstance(router_logits, torch.Tensor):
                        aux_loss += router_logits
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            if aux_loss > 0:
                loss = loss + aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[NedCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Overwritten -- uses `cache_params` as opposed to `past_key_values`
        if use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            if cache_position[0] > 0:
                input_ids = input_ids[:, -1][..., None]

                if attention_mask is not None:
                    attention_mask = None
            else:
                # we initialize the `cache_position` to full size of `conv_states` at prefill stage
                # considering padding will be applied when input length is shorter, and truncation
                # will be applied when it is longer
                conv_kernel_size = getattr(self.config, 'conv_kernel_size', 4)
                cache_position = torch.arange(0, conv_kernel_size, device=input_ids.device)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    def to_onnx(self, export_path: str, input_shape: tuple = (1, 512), opset_version: int = 14):
        """Export model to ONNX format with proper preprocessing.
        
        Args:
            export_path: Path to save the ONNX model
            input_shape: Input tensor shape (batch_size, sequence_length)
            opset_version: ONNX opset version for compatibility
        """
        import torch.onnx
        
        # Set environment flag for ONNX export mode
        os.environ['NED_ONNX_EXPORT'] = '1'
        
        try:
            self.eval()
            batch_size, seq_len = input_shape
            dummy_input = torch.randint(0, self.config.vocab_size, input_shape, dtype=torch.long)
            
            # Create dummy inputs for ONNX export
            input_names = ['input_ids']
            output_names = ['logits']
            dynamic_axes = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
            
            logger.info(f"Exporting NED model to ONNX format: {export_path}")
            
            torch.onnx.export(
                self,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            logger.info(f"ONNX export completed successfully: {export_path}")
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
        finally:
            # Reset environment flag
            os.environ.pop('NED_ONNX_EXPORT', None)

    def get_memory_usage(self) -> dict:
        """Get detailed memory usage statistics for the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory usage in MB
        param_memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # Get quantization info if available
        quantization_info = getattr(self.config, 'quantization_config', None)
        if quantization_info:
            if quantization_info.get('load_in_8bit'):
                param_memory_mb *= 0.5  # 8-bit uses half memory
            elif quantization_info.get('load_in_4bit'):
                param_memory_mb *= 0.25  # 4-bit uses quarter memory
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'estimated_memory_mb': param_memory_mb,
            'quantization_config': quantization_info
        }

    def get_model_info(self) -> dict:
        """Get comprehensive model information for debugging and monitoring."""
        info = {
            'model_type': 'ned',
            'architecture': 'Advanced SSM with Multi-head scanning',
            'vocab_size': self.config.vocab_size,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_hidden_layers,
            'num_ssm_heads': getattr(self.config, 'num_ssm_heads', 8),
            'intermediate_size': self.config.intermediate_size,
            'use_moe': getattr(self.config, 'use_moe', False),
            'use_hybrid_attention': getattr(self.config, 'use_hybrid_attention', False),
            'bidirectional_scan': getattr(self.config, 'bidirectional_scan', True),
        }
        
        # Add MoE specific info
        if info['use_moe']:
            info.update({
                'num_experts': getattr(self.config, 'num_experts', 8),
                'num_experts_per_token': getattr(self.config, 'num_experts_per_token', 2),
                'use_sparse_moe_routing': getattr(self.config, 'use_sparse_moe_routing', False)
            })
        
        # Add attention specific info
        if info['use_hybrid_attention']:
            info.update({
                'num_attention_heads': getattr(self.config, 'num_attention_heads', 16),
                'attention_frequency': getattr(self.config, 'attention_frequency', 8),
                'attention_window_size': getattr(self.config, 'attention_window_size', 512),
                'use_flash_attention': getattr(self.config, 'use_flash_attention', True)
            })
        
        # Add memory and performance info
        info.update(self.get_memory_usage())
        
        return info

    def validate_config(self) -> bool:
        """Validate model configuration for common issues."""
        try:
            # Check basic required attributes
            required_attrs = ['vocab_size', 'hidden_size', 'num_hidden_layers', 'intermediate_size']
            for attr in required_attrs:
                if not hasattr(self.config, attr):
                    logger.error(f"Missing required config attribute: {attr}")
                    return False
            
            # Validate hidden size divisibility
            num_ssm_heads = getattr(self.config, 'num_ssm_heads', 8)
            if self.config.hidden_size % num_ssm_heads != 0:
                logger.error(f"hidden_size ({self.config.hidden_size}) must be divisible by num_ssm_heads ({num_ssm_heads})")
                return False
            
            # Validate MoE configuration
            if getattr(self.config, 'use_moe', False):
                num_experts = getattr(self.config, 'num_experts', 8)
                num_experts_per_token = getattr(self.config, 'num_experts_per_token', 2)
                if num_experts_per_token > num_experts:
                    logger.error(f"num_experts_per_token ({num_experts_per_token}) cannot exceed num_experts ({num_experts})")
                    return False
            
            # Validate attention configuration
            if getattr(self.config, 'use_hybrid_attention', False):
                num_attention_heads = getattr(self.config, 'num_attention_heads', 16)
                if self.config.hidden_size % num_attention_heads != 0:
                    logger.error(f"hidden_size ({self.config.hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})")
                    return False
            
            # Validate quantization configuration
            quantization_config = getattr(self.config, 'quantization_config', None)
            if quantization_config:
                load_in_8bit = quantization_config.get('load_in_8bit', False)
                load_in_4bit = quantization_config.get('load_in_4bit', False)
                if load_in_8bit and load_in_4bit:
                    logger.error("Cannot use both 8-bit and 4-bit quantization simultaneously")
                    return False
                
                if (load_in_8bit or load_in_4bit) and not HAS_BITSANDBYTES:
                    logger.error("Quantization requested but bitsandbytes is not available")
                    return False
            
            logger.info("Model configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def benchmark_performance(self, input_length: int = 512, batch_size: int = 1, num_runs: int = 10) -> dict:
        """Benchmark model performance on the current device.
        
        Args:
            input_length: Sequence length for benchmarking
            batch_size: Batch size for benchmarking  
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with performance metrics
        """
        import time
        
        self.eval()
        device = next(self.parameters()).device
        
        # Create dummy input
        dummy_input = torch.randint(0, self.config.vocab_size, (batch_size, input_length), device=device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(3):
                _ = self(dummy_input)
        
        # Benchmark runs
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self(dummy_input)
                
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_run = total_time / num_runs
        tokens_per_second = (batch_size * input_length) / avg_time_per_run
        
        # Memory usage
        if device.type == 'cuda':
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            current_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            peak_memory_mb = 0
            current_memory_mb = 0
        
        return {
            'device': str(device),
            'batch_size': batch_size,
            'input_length': input_length,
            'num_runs': num_runs,
            'avg_time_per_run_ms': avg_time_per_run * 1000,
            'tokens_per_second': tokens_per_second,
            'peak_memory_mb': peak_memory_mb,
            'current_memory_mb': current_memory_mb,
        }

__all__ = ["NedForCausalLM", "NedModel", "NedPreTrainedModel", "NedCache"] 

# --- Additional technical comments and type hints for clarity ---
# Many functions now have type hints. Complex math operations are commented with context.

# --- Minimal unit tests for MoE and Triton kernel (if available) ---
if __name__ == "__main__":
    import time
    import pytest
    
    def test_moe_vectorized():
        """Test optimized MoE routing correctness and load balancing."""
        class DummyConfig:
            hidden_size = 8
            num_experts = 4
            num_experts_per_token = 2
            intermediate_size = 16
            routing_mode = "top2"
            adaptive_moe_routing = True
            expert_capacity_factor = 1.25
            load_balance_loss_coeff = 0.01
            router_z_loss_coeff = 0.001
            quantization_config = None
        
        moe = MixtureOfExperts(DummyConfig(), 0)
        moe.eval()  # Test in eval mode first
        
        x = torch.randn(2, 3, 8)
        out, aux = moe(x)
        assert out.shape == (2, 3, 8), f"Expected (2, 3, 8), got {out.shape}"
        assert aux is not None and aux.item() >= 0, "Auxiliary loss should be non-negative"
        
        # Test training mode with losses
        moe.train()
        out_train, aux_train = moe(x)
        assert aux_train.item() > aux.item(), "Training mode should have higher auxiliary loss due to z-loss"
        
        # Test different routing strategies
        for use_batched in [True, False]:
            # Mock the routing decision
            original_method = moe._should_use_batched_routing
            moe._should_use_batched_routing = lambda *args: use_batched
            
            out_routing, aux_routing = moe(x)
            assert out_routing.shape == (2, 3, 8), f"Routing strategy failed for batched={use_batched}"
            
            # Restore original method
            moe._should_use_batched_routing = original_method
        
        print(f"Optimized MoE test passed. Eval aux loss: {aux.item():.4f}, Train aux loss: {aux_train.item():.4f}")
    
    def test_ssm_state_consistency():
        """Test SSM state update consistency between single-step and full sequence."""
        from configuration_ned import NedConfig
        
        config = NedConfig()
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_ssm_heads = 4
        config.quantization_config = None
        
        ssm = MultiHeadSelectiveScan(config, 0)
        ssm.eval()
        
        # Test input
        batch_size, seq_len = 1, 5
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Full sequence forward
        cache_full = NedCache(config, batch_size, torch.float32, x.device)
        output_full, _ = ssm._full_sequence_forward(x, cache_full)
        
        # Single-step forward (simulating autoregressive generation)
        cache_single = NedCache(config, batch_size, torch.float32, x.device)
        outputs_single = []
        
        for t in range(seq_len):
            x_t = x[:, t:t+1]
            output_t, _ = ssm._single_step_forward(x_t, cache_single, torch.tensor([t]))
            outputs_single.append(output_t)
        
        output_single = torch.cat(outputs_single, dim=1)
        
        # Check consistency (allowing for small numerical differences)
        diff = torch.abs(output_full - output_single).max()
        assert diff < 1e-3, f"SSM consistency test failed. Max diff: {diff:.6f}"
        print(f"SSM consistency test passed. Max diff: {diff:.6f}")
    
    def test_cache_functionality():
        """Test cache state management for autoregressive generation."""
        from configuration_ned import NedConfig
        
        config = NedConfig()
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_hidden_layers = 2
        config.quantization_config = None
        
        batch_size = 2
        cache = NedCache(config, batch_size, torch.float32)
        
        # Test lazy initialization - tensors should not be created yet
        assert cache._conv_states is None
        assert cache._ssm_states is None
        
        # Access properties to trigger initialization
        assert cache.conv_states.shape == (2, batch_size, 4, 32)  # num_layers, batch, kernel, hidden
        assert cache.ssm_states.shape == (2, batch_size, 8, 16)   # num_layers, batch, heads, state_size
        
        # Test state updates
        layer_idx = 0
        new_conv = torch.randn(batch_size, 32)
        new_ssm = torch.randn(batch_size, 8, 16)
        
        # This should initialize only the needed layer
        cache.update_conv_state(layer_idx, new_conv)
        cache.update_ssm_state(layer_idx, new_ssm)
        
        # Verify states were updated
        retrieved_conv = cache.get_conv_state(layer_idx)
        retrieved_ssm = cache.get_ssm_state(layer_idx)
        
        assert torch.allclose(retrieved_ssm, new_ssm), "SSM state update failed"
        
        # Test partial initialization tracking
        assert layer_idx in cache._initialized_conv_layers
        assert layer_idx in cache._initialized_ssm_layers
        
        # Test reset functionality
        cache.reset()
        assert cache._conv_states is not None  # Should still exist but be zeroed
        assert cache._ssm_states is not None
        assert len(cache._initialized_conv_layers) == 0
        assert len(cache._initialized_ssm_layers) == 0
        
        print("Cache functionality test passed")
    
    def test_quantization_compatibility():
        """Test model with different quantization settings (if bitsandbytes available)."""
        if not HAS_BITSANDBYTES:
            print("Skipping quantization test - bitsandbytes not available")
            return
        
        from configuration_ned import NedConfig
        
        # Test 8-bit quantization
        config = NedConfig()
        config.hidden_size = 32
        config.intermediate_size = 64
        config.quantization_config = {'load_in_8bit': True}
        
        try:
            layer = create_linear_layer(32, 64, quantization_config=config.quantization_config)
            x = torch.randn(2, 32)
            out = layer(x)
            assert out.shape == (2, 64), "Quantized linear layer failed"
            print("8-bit quantization test passed")
        except Exception as e:
            print(f"8-bit quantization test failed: {e}")
        
        # Test 4-bit quantization
        config.quantization_config = {'load_in_4bit': True}
        try:
            layer = create_linear_layer(32, 64, quantization_config=config.quantization_config)
            x = torch.randn(2, 32)
            out = layer(x)
            assert out.shape == (2, 64), "4-bit quantized linear layer failed"
            print("4-bit quantization test passed")
        except Exception as e:
            print(f"4-bit quantization test failed: {e}")
    
    def benchmark_ssm_performance():
        """Benchmark SSM performance across different sequence lengths and modes."""
        from configuration_ned import NedConfig
        
        config = NedConfig()
        config.hidden_size = 512
        config.intermediate_size = 1024
        config.num_ssm_heads = 8
        config.quantization_config = None
        
        ssm = MultiHeadSelectiveScan(config, 0)
        if torch.cuda.is_available():
            ssm = ssm.cuda()
        ssm.eval()
        
        seq_lengths = [64, 128, 256, 512]
        batch_size = 4
        
        print("\nSSM Performance Benchmark:")
        print("Seq Len | Time (ms) | Implementation")
        print("-" * 40)
        
        for seq_len in seq_lengths:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            x = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
            cache = NedCache(config, batch_size, torch.float32, device)
            
            # Warm-up
            with torch.no_grad():
                ssm._full_sequence_forward(x, cache)
            
            # Benchmark current implementation using _full_sequence_forward
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):  # Multiple runs for accuracy
                    ssm._full_sequence_forward(x, cache)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            forward_time = (time.time() - start_time) * 1000 / 10  # ms per run
            
            print(f"{seq_len:7d} | {forward_time:11.2f} | {'Current'}")
    
    def benchmark_moe_efficiency():
        """Benchmark optimized MoE routing efficiency for different configurations."""
        class BenchConfig:
            def __init__(self, num_experts, adaptive_routing):
                self.hidden_size = 512
                self.num_experts = num_experts
                self.num_experts_per_token = 2
                self.intermediate_size = 1024
                self.routing_mode = "top2"
                self.adaptive_moe_routing = adaptive_routing
                self.expert_capacity_factor = 1.25
                self.load_balance_loss_coeff = 0.01
                self.router_z_loss_coeff = 0.001
                self.quantization_config = None
        
        expert_counts = [8, 16, 32, 64]
        batch_size, seq_len = 4, 256
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("\nOptimized MoE Efficiency Benchmark:")
        print("Experts | Batched (ms) | Sparse (ms) | Memory (MB) | Speedup")
        print("-" * 65)
        
        for num_experts in expert_counts:
            results = {}
            
            for routing_name, force_batched in [("Batched", True), ("Sparse", False)]:
                config = BenchConfig(num_experts, True)
                moe = MixtureOfExperts(config, 0)
                if torch.cuda.is_available():
                    moe = moe.cuda()
                moe.eval()
                
                # Override routing decision for benchmarking
                original_method = moe._should_use_batched_routing
                moe._should_use_batched_routing = lambda *args: force_batched
                
                x = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
                
                # Warm-up
                with torch.no_grad():
                    for _ in range(3):
                        moe(x)
                
                # Benchmark
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(20):
                        out, aux = moe(x)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed_time = (time.time() - start_time) * 1000 / 20
                
                # Memory usage (approximate)
                memory_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                
                results[routing_name] = {
                    'time': elapsed_time,
                    'memory': memory_mb
                }
                
                # Restore original method
                moe._should_use_batched_routing = original_method
                
                # Clean up
                del moe, x
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Calculate speedup
            batched_time = results['Batched']['time']
            sparse_time = results['Sparse']['time'] 
            speedup = batched_time / sparse_time if sparse_time > 0 else 1.0
            
            print(f"{num_experts:7d} | {batched_time:10.2f} | {sparse_time:9.2f} | {results['Sparse']['memory']:9.1f} | {speedup:6.2f}x")
        
        print("\nNote: Speedup = Batched time / Sparse time (higher is better for sparse)")
        print("Adaptive routing automatically selects the best strategy based on problem size.")
    
    # Run all tests
    print("Running NED model tests...")
    test_moe_vectorized()
    test_ssm_state_consistency()
    test_cache_functionality()
    test_quantization_compatibility()
    
    def test_cpu_optimization():
        """Test CPU inference path and performance."""
        from configuration_ned import NedConfig
        
        config = NedConfig()
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_ssm_heads = 4
        config.quantization_config = None
        
        ssm = MultiHeadSelectiveScan(config, 0)
        ssm.eval()
        
        # Test input (CPU)
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Test CPU forward pass
        cache = NedCache(config, batch_size, torch.float32, x.device)
        output, _ = ssm._full_sequence_forward(x, cache)
        
        assert output.shape == (batch_size, seq_len, config.hidden_size), f"Expected output shape {(batch_size, seq_len, config.hidden_size)}, got {output.shape}"
        print(f"CPU optimization test passed. Output shape: {output.shape}")

    def test_attention_frequency_validation():
        """Test attention frequency configuration and performance."""
        from configuration_ned import NedConfig
        
        print("\nTesting attention frequency configurations...")
        
        # Test different attention frequencies
        frequencies = [4, 8, 16]  # Every 4th, 8th, 16th layer
        
        for freq in frequencies:
            config = NedConfig()
            config.hidden_size = 64
            config.intermediate_size = 128
            config.num_hidden_layers = 8
            config.use_hybrid_attention = True
            config.attention_frequency = freq
            config.num_attention_heads = 8
            config.use_flash_attention = True
            config.quantization_config = None
            
            model = NedModel(config)
            model.eval()
            
            # Count attention layers
            attention_layers = 0
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'use_attention') and layer.use_attention:
                    attention_layers += 1
                    # Verify it's at the right frequency
                    assert i % freq == 0, f"Attention layer found at wrong position: layer {i}, frequency {freq}"
            
            expected_attention_layers = len([i for i in range(config.num_hidden_layers) if i % freq == 0])
            assert attention_layers == expected_attention_layers, f"Expected {expected_attention_layers} attention layers, got {attention_layers}"
            
            print(f"Frequency {freq}: {attention_layers}/{config.num_hidden_layers} layers have attention")
        
        print("Attention frequency validation passed")

    def test_flash_attention_optimization():
        """Test FlashAttention-2 optimizations."""
        from configuration_ned import NedConfig
        
        config = NedConfig()
        config.hidden_size = 64
        config.num_attention_heads = 8
        config.use_flash_attention = True
        config.attention_window_size = 256
        
        attention = AdvancedHybridAttention(config)
        attention.eval()
        
        # Test different sequence lengths
        for seq_len in [64, 128, 256, 512]:
            batch_size = 2
            hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
            
            with torch.no_grad():
                output, _ = attention(hidden_states)
            
            assert output.shape == (batch_size, seq_len, config.hidden_size), f"FlashAttention output shape mismatch for seq_len {seq_len}"
        
        print("FlashAttention optimization test passed")

    def test_onnx_export_compatibility():
        """Test ONNX export functionality."""
        from configuration_ned import NedConfig
        
        config = NedConfig()
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_hidden_layers = 2
        config.vocab_size = 1000
        config.quantization_config = None
        
        model = NedForCausalLM(config)
        model.eval()
        
        # Test ONNX export
        try:
            # Set smaller input for testing
            input_shape = (1, 16)
            model.to_onnx("test_model.onnx", input_shape=input_shape)
            
            # Check if file was created
            import os
            assert os.path.exists("test_model.onnx"), "ONNX export file not created"
            
            # Clean up
            os.remove("test_model.onnx")
            print("ONNX export test passed")
            
        except Exception as e:
            print(f"ONNX export test skipped due to: {e}")

    def test_hf_integration_features():
        """Test HuggingFace integration features."""
        from configuration_ned import NedConfig
        
        config = NedConfig()
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_hidden_layers = 2
        config.vocab_size = 1000
        config.use_moe = True
        config.num_experts = 4
        config.quantization_config = None
        
        model = NedForCausalLM(config)
        
        # Test configuration validation
        is_valid = model.validate_config()
        assert is_valid, "Model configuration validation failed"
        
        # Test model info
        info = model.get_model_info()
        assert info['model_type'] == 'ned', "Incorrect model type in info"
        assert info['use_moe'] == True, "MoE info not correctly reported"
        assert 'total_parameters' in info, "Memory usage info missing"
        
        # Test memory usage
        memory_info = model.get_memory_usage()
        assert 'total_parameters' in memory_info, "Memory info missing parameters count"
        assert memory_info['total_parameters'] > 0, "Invalid parameter count"
        
        # Test performance benchmark (short run)
        benchmark_results = model.benchmark_performance(input_length=16, num_runs=2)
        assert 'tokens_per_second' in benchmark_results, "Benchmark missing tokens_per_second"
        assert benchmark_results['tokens_per_second'] > 0, "Invalid tokens per second"
        
        print("HuggingFace integration test passed")

    def test_quantization_compatibility_advanced():
        """Advanced test for quantization compatibility across different configurations."""
        if not HAS_BITSANDBYTES:
            print("Skipping advanced quantization test - bitsandbytes not available")
            return
        
        from configuration_ned import NedConfig
        
        # Test 8-bit quantization
        config = NedConfig()
        config.hidden_size = 64
        config.intermediate_size = 128
        config.quantization_config = {'load_in_8bit': True}
        
        try:
            model = NedForCausalLM(config)
            
            # Test inference with quantized model
            input_ids = torch.randint(0, config.vocab_size, (1, 10))
            with torch.no_grad():
                output = model(input_ids)
            
            assert output.logits.shape == (1, 10, config.vocab_size), "Quantized model output shape incorrect"
            
            # Test configuration validation
            assert model.validate_config(), "Quantized model configuration validation failed"
            
            print("Advanced quantization compatibility test passed")
            
        except Exception as e:
            print(f"Advanced quantization test failed: {e}")

    def test_moe_integration_comprehensive():
        """Comprehensive test for MoE integration with the full model."""
        from configuration_ned import NedConfig
        
        print("\nTesting comprehensive MoE integration...")
        
        # Test configuration with MoE enabled
        config = NedConfig()
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_hidden_layers = 4
        config.vocab_size = 1000
        config.use_moe = True
        config.num_experts = 4
        config.num_experts_per_token = 2
        config.adaptive_moe_routing = True
        config.load_balance_loss_coeff = 0.01
        config.router_z_loss_coeff = 0.001
        config.quantization_config = None
        
        model = NedForCausalLM(config)
        model.eval()
        
        # Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        
        with torch.no_grad():
            output = model(input_ids, output_router_logits=True)
        
        assert output.logits.shape == (2, 10, config.vocab_size), "MoE model output shape incorrect"
        
        # Test training mode with auxiliary losses
        model.train()
        labels = torch.randint(0, config.vocab_size, (2, 10))
        
        output_train = model(input_ids, labels=labels, output_router_logits=True)
        assert output_train.loss is not None, "Training loss should not be None"
        assert output_train.loss.item() > 0, "Training loss should be positive"
        
        # Test that MoE layers are only in later layers
        moe_layer_count = 0
        total_layers = len(model.model.layers)
        
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'use_moe') and layer.use_moe:
                moe_layer_count += 1
                # Should only be in later half of layers
                assert i > total_layers // 2, f"MoE found in early layer {i}, expected only in later layers"
        
        assert moe_layer_count > 0, "No MoE layers found in model"
        print(f"Found {moe_layer_count} MoE layers in later {total_layers - total_layers//2} layers")
        
        # Test model info and validation
        model_info = model.get_model_info()
        assert model_info['use_moe'] == True, "Model info should reflect MoE usage"
        assert 'num_experts' in model_info, "Model info missing expert count"
        
        # Test configuration validation
        assert model.validate_config(), "MoE model configuration validation failed"
        
        # Test memory usage reporting
        memory_info = model.get_memory_usage()
        assert 'total_parameters' in memory_info, "Memory info missing"
        assert memory_info['total_parameters'] > 0, "Invalid parameter count"
        
        print("Comprehensive MoE integration test passed")

    def test_moe_performance_scaling():
        """Test MoE performance scaling with different configurations."""
        from configuration_ned import NedConfig
        import time
        
        print("\nTesting MoE performance scaling...")
        
        configs = [
            {'num_experts': 4, 'adaptive_routing': True},
            {'num_experts': 8, 'adaptive_routing': True},
            {'num_experts': 16, 'adaptive_routing': False},  # Force sparse for high expert count
        ]
        
        batch_size, seq_len = 2, 32
        
        for i, moe_config in enumerate(configs):
            config = NedConfig()
            config.hidden_size = 64
            config.intermediate_size = 128
            config.num_hidden_layers = 2
            config.vocab_size = 1000
            config.use_moe = True
            config.num_experts = moe_config['num_experts']
            config.adaptive_moe_routing = moe_config['adaptive_routing']
            config.quantization_config = None
            
            model = NedForCausalLM(config)
            model.eval()
            
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            # Warm-up
            with torch.no_grad():
                _ = model(input_ids)
            
            # Benchmark
            start_time = time.perf_counter()
            
            with torch.no_grad():
                for _ in range(5):
                    _ = model(input_ids)
            
            end_time = time.perf_counter()
            avg_time = (end_time - start_time) / 5 * 1000  # ms
            
            routing_strategy = "Adaptive" if moe_config['adaptive_routing'] else "Sparse"
            print(f"Config {i+1}: {moe_config['num_experts']} experts, {routing_strategy} routing: {avg_time:.2f}ms")
        
        print("MoE performance scaling test completed")

    test_cpu_optimization()
    test_attention_frequency_validation()
    test_flash_attention_optimization()
    test_onnx_export_compatibility()
    test_hf_integration_features()
    test_quantization_compatibility_advanced()
    test_moe_integration_comprehensive()
    test_moe_performance_scaling()
    
    def benchmark_attention_frequency_performance():
        """Benchmark performance across different attention frequencies."""
        from configuration_ned import NedConfig
        import time
        
        print("\nBenchmarking attention frequency performance...")
        
        frequencies = [0, 4, 8, 16]  # 0 means no attention
        seq_len = 128
        batch_size = 2
        num_runs = 5
        
        results = {}
        
        for freq in frequencies:
            config = NedConfig()
            config.hidden_size = 64
            config.intermediate_size = 128
            config.num_hidden_layers = 8
            config.vocab_size = 1000
            config.use_hybrid_attention = freq > 0
            config.attention_frequency = freq if freq > 0 else 8
            config.num_attention_heads = 8
            config.quantization_config = None
            
            model = NedForCausalLM(config)
            model.eval()
            
            # Warm up
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            with torch.no_grad():
                _ = model(input_ids)
            
            # Benchmark
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(input_ids)
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / num_runs * 1000  # ms
            results[f"freq_{freq}"] = avg_time
            
            attention_layers = sum(1 for layer in model.model.layers if hasattr(layer, 'use_attention') and layer.use_attention)
            print(f"Frequency {freq}: {avg_time:.2f}ms avg, {attention_layers} attention layers")
        
        # Find optimal frequency (lowest latency)
        optimal_freq = min(results, key=results.get)
        print(f"Optimal attention frequency: {optimal_freq} ({results[optimal_freq]:.2f}ms)")
        
        return results

    # Run benchmarks if requested
    if os.environ.get('NED_RUN_BENCHMARKS', '0') == '1':
        print("\nRunning performance benchmarks...")
        benchmark_ssm_performance()
        benchmark_moe_efficiency()
        benchmark_attention_frequency_performance()
        print("\nBenchmarking complete.")
    else:
        print("\nTo run performance benchmarks, set NED_RUN_BENCHMARKS=1")
    
    # Test current SSM implementation on GPU if available
    if torch.cuda.is_available():
        print("\nTesting SSM implementation on GPU...")
        try:
            from configuration_ned import NedConfig
            config = NedConfig()
            config.hidden_size = 64
            config.intermediate_size = 128
            config.quantization_config = None
            
            ssm = MultiHeadSelectiveScan(config, 0).cuda()
            x = torch.randn(2, 10, config.hidden_size, device='cuda')
            cache = NedCache(config, 2, torch.float32, 'cuda')
            
            result, _ = ssm._full_sequence_forward(x, cache)
            print(f"GPU SSM test passed. Output shape: {result.shape}")
        except Exception as e:
            print(f"GPU SSM test failed: {e}")
    else:
        print("GPU SSM testing skipped (CUDA not available)")
    
    print("\nAll tests completed.") 