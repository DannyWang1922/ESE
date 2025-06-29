"""PyTorch Qwen3 MoE model."""

import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
import numpy as np

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RotaryEmbedding,
    Qwen3PreTrainedModel,
    Qwen3DecoderLayer,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
    QWEN3_START_DOCSTRING,
    QWEN3_INPUTS_DOCSTRING,
)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen3-8B"
_CONFIG_FOR_DOC = "Qwen3Config"


class Qwen3MoEExpert(nn.Module):
    """
    Expert module for Mixture of Experts Qwen3 model.
    
    Each expert is a feed-forward network with a bottleneck architecture,
    consisting of two down-projection layers.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create expert's down-projection layers
        self.down_proj1 = nn.Linear(config.hidden_size, config.moe_expert_intermediate_size, bias=False)
        self.down_proj2 = nn.Linear(config.moe_expert_intermediate_size, config.moe_expert_compressed_size, bias=False)
        
        # Activation function (matching Qwen3's activation)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act
        
        # Initialize expert weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize the expert weights using Qwen3's initialization strategy.
        """
        # Initialize down-projection layers
        nn.init.normal_(self.down_proj1.weight, mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(self.down_proj2.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(self, hidden_states):
        """
        Forward pass through the expert.
        
        Args:
            hidden_states: Tensor of shape [batch_size * seq_len, hidden_size]
                Token representations to process.
                
        Returns:
            output: Tensor of shape [batch_size * seq_len, moe_expert_compressed_size]
                Processed token representations.
        """
        # Forward through down-projection layers with activation
        intermediate_output = self.act_fn(self.down_proj1(hidden_states))
        output = self.down_proj2(intermediate_output)
        
        return output


class Qwen3MoEGate(nn.Module):
    """
    Gate module for Mixture of Experts Qwen3 model.
    
    This module determines which experts should process each token by computing
    routing probabilities.
    """
    
    def __init__(self, config, num_experts=8, top_k=2):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gate projection to calculate expert routing logits
        self.gate_weights = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        
        # Initialize gate weights
        nn.init.kaiming_uniform_(self.gate_weights.weight, a=math.sqrt(5))
        
        # Optional: add noise to encourage exploration
        self.noise_epsilon = getattr(config, 'router_noise_epsilon', 1e-2)
        self.training_noise = getattr(config, 'router_training_noise', True)
        
        # Temperature for softmax
        self.temperature = getattr(config, 'router_temperature', 1.0)
        
        # Load balancing parameters
        self.use_load_balancing = getattr(config, 'use_load_balancing', False)
        self.router_z_loss_coef = getattr(config, 'router_z_loss_coef', 1e-3)
        self.router_aux_loss_coef = getattr(config, 'router_aux_loss_coef', 0.01)

    def forward(self, hidden_states):
        """
        Calculate routing probabilities for each token to each expert.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
                Token representations from which to compute routing probabilities.
                
        Returns:
            gate_logits: Tensor of shape [batch_size, seq_len, num_experts]
                Logits for routing each token to each expert.
        """
        # Calculate gate logits
        gate_logits = self.gate_weights(hidden_states)
        
        # Add noise during training to encourage exploration
        if self.training and self.training_noise and self.noise_epsilon > 0:
            gate_noise = torch.randn_like(gate_logits) * self.noise_epsilon
            gate_logits = gate_logits + gate_noise
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            gate_logits = gate_logits / self.temperature
            
        return gate_logits


class Qwen3MoEBlock(nn.Module):
    """
    MoE Block that combines gating mechanism and expert pool.
    This module supplements the FFN layer in transformer blocks.
    """
    
    def __init__(self, config, num_experts=8, top_k=2):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create the gate for selecting experts
        self.gate = Qwen3MoEGate(config, num_experts, top_k)
        
        # Create pool of experts
        self.experts = nn.ModuleList([Qwen3MoEExpert(config) for _ in range(num_experts)])
        
        # Expert initialization strategy
        self.expert_init_strategy = getattr(config, 'expert_init_strategy', 'identical')
        
        # Apply initialization strategy
        if self.expert_init_strategy == "diverse":
            self._initialize_diverse_experts()
        
        # Optional expert dropout
        self.expert_dropout = getattr(config, 'expert_dropout', 0.0)
        
        # Parallel computation flag
        self.parallel_computation = getattr(config, 'parallel_expert_computation', True)
        
        # Metrics tracking
        self.expert_metrics = {
            "expert_utilization": [0.0] * self.num_experts,
            "expert_load_balance": 0.0,
        }
    
    def _initialize_diverse_experts(self):
        """
        Initialize experts with diverse parameters to encourage specialization.
        """
        for i, expert in enumerate(self.experts):
            with torch.no_grad():
                # Scale weights by a small factor based on expert index
                scale_factor = 1.0 + (i - self.num_experts // 2) * 0.01
                expert.down_proj1.weight.data *= scale_factor
    
    def forward(self, hidden_states):
        """
        Forward pass through MoE block.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
                Input hidden states
                
        Returns:
            output: Tensor of shape [batch_size, seq_len, moe_expert_compressed_size]
                MoE processed output
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get gating logits and routing decisions
        router_logits = self.gate(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)
        routing_weights, selected_experts_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Store for potential load balancing loss
        self._last_gate_logits = router_logits
        self._last_gate_indices = selected_experts_indices
        
        # Initialize output tensor
        moe_output = torch.zeros(
            batch_size, seq_len, self.config.moe_expert_compressed_size,
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        
        # Track expert usage
        expert_counts = torch.zeros(self.num_experts, device=hidden_states.device)
        
        # Reshape hidden states for expert processing
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Process tokens through selected experts
        for expert_idx in range(self.num_experts):
            # Find which tokens are routed to this expert
            expert_mask = (selected_experts_indices == expert_idx)
            
            if expert_mask.any():
                # Count usage
                expert_counts[expert_idx] = expert_mask.sum().item()
                
                # Get indices and probabilities
                batch_indices, seq_indices, k_indices = expert_mask.nonzero(as_tuple=True)
                
                # Get flat indices for gathering
                flat_indices = batch_indices * seq_len + seq_indices
                
                # Get the corresponding gate probabilities
                token_probs = routing_weights[batch_indices, seq_indices, k_indices]
                
                # Get the hidden states for these tokens
                token_hidden_states = hidden_states_flat[flat_indices]
                
                # Forward through the expert
                expert_output = self.experts[expert_idx](token_hidden_states)
                
                # Scale by gate probabilities
                scaled_expert_output = expert_output * token_probs.unsqueeze(-1)
                
                # Accumulate the expert outputs
                moe_output[batch_indices, seq_indices] += scaled_expert_output
        
        # Update expert utilization metrics
        total_tokens = batch_size * seq_len * self.top_k
        expert_utilization_list = []
        for i in range(self.num_experts):
            utilization = float(expert_counts[i].item()) / total_tokens
            expert_utilization_list.append(utilization)
        self.expert_metrics["expert_utilization"] = expert_utilization_list
        
        # Compute load balance metric
        expert_utilization_tensor = torch.tensor(expert_utilization_list)
        if torch.mean(expert_utilization_tensor) > 0:
            load_balance = torch.std(expert_utilization_tensor) / torch.mean(expert_utilization_tensor)
            self.expert_metrics["expert_load_balance"] = float(load_balance.item())
        else:
            self.expert_metrics["expert_load_balance"] = 0.0
        
        return moe_output


class Qwen3LayerWithMoEBlock(nn.Module):
    """
    Qwen3 decoder layer with MoE block integrated.
    The MoE block supplements the standard FFN computation.
    """
    
    def __init__(self, config: Qwen3Config, layer_idx: int, num_experts=8, top_k=2):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MoE components
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_block = Qwen3MoEBlock(config, num_experts, top_k)
        
        # Expert metrics
        self.expert_metrics = {
            "expert_utilization": [0.0] * self.num_experts,
            "expert_load_balance": 0.0
        }
        self._cached_moe_output = None
        
        # Sliding window warning (from original Qwen3)
        if (
            config.sliding_window and config._attn_implementation != "flash_attention_2"
        ):  # diff with Llama is this warning
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Standard MLP
        mlp_output = self.mlp(hidden_states)
        
        # MoE computation (on normalized hidden states)
        moe_output = self.moe_block(hidden_states)
        
        # Cache MoE output for potential access
        self._cached_moe_output = moe_output
        
        # Update expert metrics
        self.expert_metrics["expert_utilization"] = self.moe_block.expert_metrics["expert_utilization"].copy()
        self.expert_metrics["expert_load_balance"] = self.moe_block.expert_metrics["expert_load_balance"]
        
        # Combine MLP output with residual (MoE output is handled separately)
        hidden_states = residual + mlp_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
    
    def get_cached_moe_output(self, clear_cache=False):
        """Get the cached MoE output."""
        moe_output = self._cached_moe_output
        if clear_cache:
            self._cached_moe_output = None
        return moe_output


class Qwen3MoEModel(Qwen3PreTrainedModel):
    """
    The bare Qwen3 MoE Model outputting raw hidden-states without any specific head on top.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Store MoE parameters
        self.num_experts = getattr(config, 'num_experts', 8)
        self.top_k = getattr(config, 'top_k', 2)
        
        # Get MoE layer configuration
        self.moe_layers = getattr(config, 'moe_layers', 'all')
        
        # Convert moe_layers to a set of indices
        if self.moe_layers == 'all':
            self.moe_layer_indices = set(range(config.num_hidden_layers))
        elif isinstance(self.moe_layers, (list, tuple)):
            self.moe_layer_indices = set(self.moe_layers)
        else:
            raise ValueError(f"Invalid moe_layers format: {self.moe_layers}")
        
        # Validate layer indices
        for idx in self.moe_layer_indices:
            if not isinstance(idx, int) or idx < 0 or idx >= config.num_hidden_layers:
                raise ValueError(f"Invalid layer index {idx}")

        # Embeddings and normalization
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        
        # Create layers with mixed MoE and standard layers
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in self.moe_layer_indices:
                self.layers.append(
                    Qwen3LayerWithMoEBlock(config, layer_idx, self.num_experts, self.top_k)
                )
            else:
                self.layers.append(Qwen3DecoderLayer(config, layer_idx))
        
        self.gradient_checkpointing = False
        
        # Expert metrics tracking
        self.track_expert_metrics = getattr(config, 'track_expert_metrics', True)
        if self.track_expert_metrics:
            self.expert_metrics = {
                "expert_utilization": [0.0] * self.num_experts,
                "expert_load_balance": 0.0
            }

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def collect_moe_outputs(self):
        """Collect MoE outputs from all layers that have them."""
        moe_outputs = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Qwen3LayerWithMoEBlock) and hasattr(layer, 'get_cached_moe_output'):
                moe_output = layer.get_cached_moe_output(clear_cache=False)
                if moe_output is not None:
                    moe_outputs[i] = moe_output
        return moe_outputs

    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # Legacy cache handling
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Expert metrics tracking initialization
        if self.track_expert_metrics:
            self.expert_metrics["per_layer_utilization"] = []
            self.expert_metrics["per_layer_expert_load_balance"] = []
            self.expert_metrics["expert_utilization"] = None
            self.expert_metrics["expert_load_balance"] = 0.0
            moe_layers_count = 0

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers[:self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            # Track expert metrics for MoE layers
            if self.track_expert_metrics and layer_idx in self.moe_layer_indices:
                moe_layers_count += 1
                if hasattr(decoder_layer, "expert_metrics"):
                    layer_util = decoder_layer.expert_metrics.get("expert_utilization")
                    if layer_util is not None:
                        self.expert_metrics["per_layer_utilization"].append((layer_idx, layer_util.copy()))

                    layer_balance = decoder_layer.expert_metrics.get("expert_load_balance")
                    if layer_balance is not None:
                        self.expert_metrics["per_layer_expert_load_balance"].append((layer_idx, layer_balance))

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Calculate final expert utilization metrics
        if self.track_expert_metrics and moe_layers_count > 0:
            if self.expert_metrics["per_layer_utilization"]:
                all_utils = np.array([util for _, util in self.expert_metrics["per_layer_utilization"]])
                avg_util = np.mean(all_utils, axis=0).tolist()
                self.expert_metrics["expert_utilization"] = avg_util
            
            if self.expert_metrics["per_layer_expert_load_balance"]:
                load_balance_list = [balance for _, balance in self.expert_metrics["per_layer_expert_load_balance"]]
                self.expert_metrics["expert_load_balance"] = sum(load_balance_list) / len(load_balance_list)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen3. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen3Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)`.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class Qwen3MoEForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """
    Qwen3 MoE Model transformer with a language modeling head on top.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3MoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        loss_kwargs: Optional[LossKwargs] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Give me a short introduction to large language model."
        >>> messages = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}]
        >>> text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        >>> generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        >>> generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        >>> response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **loss_kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3MoEForSequenceClassification(Qwen3PreTrainedModel):
    """
    The Qwen3 MoE Model transformer with a sequence classification head on top.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen3MoEModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # Handle both left- and right- padding
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )