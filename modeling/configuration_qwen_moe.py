from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3MoEConfig(Qwen3Config):
    """
    Configuration class for Qwen3MoE model.
    
    This extends the original Qwen3Config with parameters specific to the
    Mixture of Experts architecture.
    """
    
    model_type = "qwen3_moe"
    
    def __init__(
        self,
        # MoE specific parameters
        num_experts=8,
        top_k=2,
        expert_dropout=0.0,
        expert_init_strategy="identical",  # or "diverse"
        router_temperature=1.0,
        router_noise_epsilon=1e-2,
        router_training_noise=True,
        use_load_balancing=False,
        router_z_loss_coef=1e-3,
        router_aux_loss_coef=0.01,
        track_expert_metrics=True,
        parallel_expert_computation=True,
        moe_layers="all",  # "all" or list of layer indices like [0, 4, 8, 12]
        moe_expert_intermediate_size=512,
        moe_expert_compressed_size=256,
        **kwargs
    ):
        """
        Initialize Qwen3MoEConfig with MoE-specific parameters.
        
        Args:
            # MoE specific parameters
            num_experts: Number of expert feed-forward networks per layer.
            top_k: Number of experts to route each token to.
            expert_dropout: Probability of dropping out entire experts during training.
            expert_init_strategy: Strategy for initializing experts ("identical" or "diverse").
            router_temperature: Temperature for router softmax to control sharpness.
            router_noise_epsilon: Magnitude of noise to add to router logits during training.
            router_training_noise: Whether to add noise to router logits during training.
            use_load_balancing: Whether to use auxiliary load balancing loss.
            router_z_loss_coef: Coefficient for router z-loss to improve stability.
            router_aux_loss_coef: Coefficient for router auxiliary losses.
            track_expert_metrics: Whether to track and log expert utilization metrics.
            parallel_expert_computation: Whether to compute expert outputs in parallel.
            moe_layers: Which layers to apply MoE to. Can be "all" or a list of layer indices.
            moe_expert_intermediate_size: Size of the intermediate layer in each expert.
            moe_expert_compressed_size: Size of the compressed output layer in each expert.
        """
        super().__init__(**kwargs)
        
        # Store MoE specific parameters
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dropout = expert_dropout
        self.expert_init_strategy = expert_init_strategy
        self.router_temperature = router_temperature
        self.router_noise_epsilon = router_noise_epsilon
        self.router_training_noise = router_training_noise
        self.use_load_balancing = use_load_balancing
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef
        self.track_expert_metrics = track_expert_metrics
        self.parallel_expert_computation = parallel_expert_computation
        self.moe_layers = moe_layers
        self.moe_expert_intermediate_size = moe_expert_intermediate_size
        self.moe_expert_compressed_size = moe_expert_compressed_size