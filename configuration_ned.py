"""Configuration class for NED model with comprehensive optimization parameters."""

import math
import torch
from transformers import PretrainedConfig


class NedConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`NedModel`]. It is used to instantiate a NED
    model according to the specified arguments, defining the model architecture with advanced SSM optimizations.

    Args:
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the NED model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        state_size (`int`, *optional*, defaults to 128):
            Shape of the state space latents.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimensionality of the inner feed-forward layers.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end of sentence token in the vocabulary.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        hidden_act (`str`, *optional*, defaults to "silu"):
            The non-linear activation function (function or string) in the decoder.
        use_cache (`bool`, *optional*, defaults to True):
            Whether or not the model should return the last state/key/value.
        tie_word_embeddings (`bool`, *optional*, defaults to False):
            Whether or not to tie the word embeddings with the input embeddings.
        rms_norm (`bool`, *optional*, defaults to True):
            Whether to use RMS normalization.
        use_bias (`bool`, *optional*, defaults to False):
            Whether or not to use bias in linear layers.
        conv_kernel (`int`, *optional*, defaults to 4):
            Size of the convolution kernel in SSM blocks.
        expand (`int`, *optional*, defaults to 2):
            Expansion factor for intermediate size.
        time_step_rank (`Union[int,str]`, *optional*, defaults to "auto"):
            Rank of the time step projection.
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum time step value.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum time step value.
        time_step_floor (`float`, *optional*, defaults to 1e-4):
            Floor for time step values.
        residual_in_fp32 (`bool`, *optional*, defaults to True):
            Whether residuals should be in fp32.
        use_gradient_checkpointing (`bool`, *optional*, defaults to True):
            Whether to use gradient checkpointing.

    Example:
        >>> from transformers import NedConfig, NedModel
        >>> configuration = NedConfig()
        >>> model = NedModel(configuration)
    """

    model_type = "ned"
    attribute_map = {"max_position_embeddings": "max_seq_length", "hidden_size": "n_embd"}

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=4096,
        state_size=128,
        num_hidden_layers=32,
        intermediate_size=11008,
        max_position_embeddings=4096,
        layer_norm_eps=1e-5,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        hidden_act="silu",
        use_cache=True,
        tie_word_embeddings=False,
        rms_norm=True,
        use_bias=False,
        conv_kernel=4,
        expand=2,
        time_step_rank="auto",
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        dt_init="random",  # "random" or "constant" initialization
        dt_scale=1.0,      # Scale factor for dt bias initialization
        dt_init_floor=1e-4,  # Floor value for dt initialization
        residual_in_fp32=True,
        dropout_rate=0.1,
        num_attention_heads=8,
        num_ssm_heads=8,
        rotary_base=10000,
        conv_kernel_size=4,  # Renamed for clarity
        # Advanced SSM parameters
        bidirectional_scan=True,
        use_triton_kernels=True,
        
        # Hybrid attention parameters
        attention_frequency=4,  # Use attention every N layers
        attention_window_size=512,
        use_rope=True,
        use_alibi_fallback=False,
        
        # MoE parameters
        use_moe=False,
        num_experts=8,
        num_experts_per_token=2,
        routing_mode="top2",  # "top1", "top2", "top_k"
        adaptive_moe_routing=True,  # Dynamic routing strategy
        expert_capacity_factor=1.25,  # Expert capacity for load balancing
        load_balance_loss_coeff=0.01,  # Load balancing loss coefficient
        router_z_loss_coeff=0.001,  # Router z-loss for stability
        
        # Advanced optimization parameters
        use_layer_scale=True,
        layer_scale_init=1e-4,
        init_method_std=0.02,  # Std for parameter initialization
        use_gradient_checkpointing=True,
        
        # Enhanced normalization
        learnable_eps=False,
        center_norm=False,
        
        # Progressive architecture scaling
        progressive_layer_drop=0.0,
        adaptive_layer_depth=False,
        
        **kwargs,
    ):
        # Initialization settings
        # Define initialization methods inline to avoid external dependencies
        def init_method_normal(std):
            def init_method(tensor):
                return torch.nn.init.normal_(tensor, mean=0.0, std=std)
            return init_method
            
        def scaled_init_method_normal(std, num_layers):
            def init_method(tensor):
                return torch.nn.init.normal_(tensor, mean=0.0, std=std/math.sqrt(2*num_layers))
            return init_method
            
        self.init_method_std = init_method_std
         # Default weight initialization: normal and scaled residual
        self.init_method = init_method_normal(self.init_method_std)
        self.output_layer_init_method = scaled_init_method_normal(self.init_method_std, num_hidden_layers)
         
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.hidden_act = hidden_act
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rms_norm = rms_norm
        self.use_bias = use_bias
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.time_step_rank = math.ceil(hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.residual_in_fp32 = residual_in_fp32
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.dropout_rate = dropout_rate
        self.num_attention_heads = num_attention_heads
        self.num_ssm_heads = num_ssm_heads
        self.rotary_base = rotary_base
        self.conv_kernel_size = conv_kernel_size or conv_kernel  # Backward compat
        
        # Advanced SSM settings
        self.num_ssm_heads = num_ssm_heads
        self.bidirectional_scan = bidirectional_scan
        self.use_triton_kernels = use_triton_kernels
        
        # Hybrid attention settings
        self.attention_frequency = attention_frequency
        self.attention_window_size = attention_window_size
        self.use_rope = use_rope
        self.use_alibi_fallback = use_alibi_fallback
        
        # MoE settings
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.routing_mode = routing_mode
        self.adaptive_moe_routing = adaptive_moe_routing
        self.expert_capacity_factor = expert_capacity_factor
        self.load_balance_loss_coeff = load_balance_loss_coeff
        self.router_z_loss_coeff = router_z_loss_coeff
        
        # BlackMamba compatibility attributes
        self.num_moe_experts = num_experts  # Alias for consistency
        self.mamba_moe_layers = getattr(kwargs, 'mamba_moe_layers', None)
        self.ffn_hidden_size = intermediate_size  # MLP expects this name
        self.add_bias_linear = use_bias  # BlackMamba uses this name
        self.device = kwargs.get('device', 'cuda')
        self.gated_linear_unit = kwargs.get('gated_linear_unit', False)
        self.activation_func = kwargs.get('activation_func', torch.nn.functional.silu)
        
        # Optimization settings
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = layer_scale_init
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Enhanced normalization
        self.learnable_eps = learnable_eps
        self.center_norm = center_norm
        
        # Progressive architecture
        self.progressive_layer_drop = progressive_layer_drop
        self.adaptive_layer_depth = adaptive_layer_depth
        
        super().__init__(**kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary with additional metadata.
        """
        output = super().to_dict()
        output["model_version"] = "advanced_ssm_v2"
        output["architecture_type"] = "hybrid_ssm_moe"
        output["optimization_features"] = {
            "ssm_core": True,
            "multi_head_ssm": self.num_ssm_heads > 1,
            "bidirectional_scan": self.bidirectional_scan,
            "hybrid_attention": self.attention_frequency > 0,
            "mixture_of_experts": self.use_moe,
            "triton_kernels": self.use_triton_kernels,
            "layer_scaling": self.use_layer_scale,
            "gradient_checkpointing": self.use_gradient_checkpointing,
        }
        output["dropout_rate"] = self.dropout_rate
        return output
    
    @classmethod
    def from_pretrained_model_name(cls, model_name: str, **kwargs):
        """
        Create configuration from a pretrained model name with optimized defaults.
        """
        base_configs = {
            "ned-small": {
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "intermediate_size": 2048,
                "num_ssm_heads": 4,
                "use_moe": False,
            },
            "ned-base": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "intermediate_size": 4096,
                "num_ssm_heads": 8,
                "use_moe": False,
            },
            "ned-large": {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "intermediate_size": 8192,
                "num_ssm_heads": 16,
                "use_moe": True,
                "num_experts": 16,
            },
            "ned-xl": {
                "hidden_size": 6144,
                "num_hidden_layers": 48,
                "intermediate_size": 12288,
                "num_ssm_heads": 24,
                "use_moe": True,
                "num_experts": 32,
                "attention_frequency": 6,
            },
        }
        
        if model_name in base_configs:
            config_dict = base_configs[model_name]
            config_dict.update(kwargs)
            return cls(**config_dict)
        else:
            return cls(**kwargs)
