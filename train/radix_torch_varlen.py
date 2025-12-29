"""
RadixMLP-enabled Qwen3ForCausalLM with variable length support.

This module implements a Qwen3 model with RadixMLP prefix-based computation sharing
and variable length sequence support using flash attention.

Design principles based on Rust ground truth:
- Batchless architecture: All tensors are num_tokens long (no batch dimension)
- use_flash_attn_varlen is always true
- RadixMLP folding/scattering for prefix sharing following Rust implementation
- Variable length sequence support with cu_seq_lengths
- Uses torch.index_select for efficient fold/scatter operations
"""

from typing import Optional, Union, Tuple, List, Any
import math

import torch
from torch import nn
from torch.nn import functional as F

# Import radix MLP functionality
try:
    from radix_mlp.torch import compute_fold_and_scatter_torch

    RADIX_MLP_AVAILABLE = True
except ImportError:
    RADIX_MLP_AVAILABLE = False
    compute_fold_and_scatter_torch = None
    print("Warning: radix_mlp not available. Using standard implementation.")

# Import flash attention functionality
try:
    from flash_attn import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_varlen_func = None
    print("Warning: flash_attn not available. Using standard attention.")


class Qwen3Config:
    """Basic Qwen3 config for compatibility."""

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 1536,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 2,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        sliding_window: Optional[int] = None,
        layer_types: Optional[List[str]] = None,
        _attn_implementation: str = "flash_attention_2",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.layer_types = layer_types or ["full_attention"] * num_hidden_layers
        self._attn_implementation = "flash_attention_2"  # Force flash attention
        self.head_dim = hidden_size // num_attention_heads

        # Set rope_parameters for compatibility
        self.rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}


class Qwen3RMSNorm(nn.Module):
    """RMS Normalization layer."""

    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embeddings to a single tensor."""
    # x: [num_tokens, num_heads, head_dim]
    # cos, sin: [num_tokens, head_dim//2]

    head_dim = x.shape[-1]
    rot_dim = cos.shape[-1] * 2

    # Split into two halves
    x1 = x[..., : rot_dim // 2]
    x2 = x[..., rot_dim // 2 : rot_dim]

    # Apply rotation
    x_rot = torch.cat([-x2, x1], dim=-1) if rot_dim == head_dim else torch.cat([x1, x2], dim=-1)

    # Apply cos/sin
    cos_expanded = cos.unsqueeze(-2)  # [num_tokens, 1, head_dim//2]
    sin_expanded = sin.unsqueeze(-2)  # [num_tokens, 1, head_dim//2]

    # Apply rotation with cos/sin
    x_rotated = x * cos_expanded + x_rot * sin_expanded

    return x_rotated


class Qwen3RotaryEmbedding(nn.Module):
    """Rotary Position Embedding layer."""

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = "default"

        base = config.rope_theta
        dim = config.head_dim
        attention_factor = 1.0

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                / dim
            )
        )
        self.attention_scaling = attention_factor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)


def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate cos and sin for given position IDs.

    Args:
        position_ids: [num_tokens] position IDs

    Returns:
        Tuple of (cos, sin) each [num_tokens, head_dim//2]
    """
    inv_freq_expanded = (
        self.inv_freq.unsqueeze(0).float().expand(position_ids.shape[0], -1).to(position_ids.device)
    )
    position_ids_expanded = position_ids.float().unsqueeze(-1)

    device_type = (
        position_ids.device.type
        if isinstance(position_ids.device.type, str) and position_ids.device.type != "mps"
        else "cpu"
    )
    with torch.cuda.amp.autocast(enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).squeeze(
            -1
        )  # [num_tokens, head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [num_tokens, head_dim]
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=position_ids.dtype), sin.to(dtype=position_ids.dtype)


# Activation function mapping
ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
    "tanh": torch.tanh,
}


class RadixMLPQwen3Config(Qwen3Config):
    """Extended Qwen3Config with RadixMLP-specific parameters."""

    def __init__(
        self,
        use_radix_mlp: bool = True,
        radix_pad_multiple_of: Optional[int] = 8,
        radix_min_prefix_length: int = 4,
        use_flash_attn_varlen: bool = True,  # Always true
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_radix_mlp = use_radix_mlp
        self.radix_pad_multiple_of = radix_pad_multiple_of
        self.radix_min_prefix_length = radix_min_prefix_length
        self.use_flash_attn_varlen = True  # Force true


class RadixMLPQwen3MLP(nn.Module):
    """RadixMLP-enabled MLP layer with folding/scattering support."""

    def __init__(self, config: RadixMLPQwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.use_radix_mlp = config.use_radix_mlp and RADIX_MLP_AVAILABLE

        # Standard MLP layers
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        x: torch.Tensor,
        fold_gather: Optional[torch.Tensor] = None,
        scatter_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional radix folding/scattering.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            fold_gather: Optional indices to gather from original to compact space
            scatter_indices: Optional indices to scatter from compact to original space

        Returns:
            Output tensor with same shape as input
        """
        if not self.use_radix_mlp or fold_gather is None or scatter_indices is None:
            # Standard MLP forward pass
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            return down_proj

        # RadixMLP-enabled forward pass following Rust ground truth
        # 1. Fold: gather from original space to compact space
        # compact[j] = original[fold_gather[j]]
        x_compact = torch.index_select(x, dim=0, index=fold_gather)  # [compact_len, hidden_size]

        # 2. Apply MLP in compact space
        gate_compact = self.act_fn(self.gate_proj(x_compact))
        up_compact = self.up_proj(x_compact)
        down_compact = self.down_proj(gate_compact * up_compact)

        # 3. Scatter: expand back to original space
        # unfolded[i] = compact[scatter_indices[i]]
        down_flat = torch.index_select(
            down_compact, dim=0, index=scatter_indices
        )  # [original_len, hidden_size]

        return down_flat


class RadixMLPQwen3Attention(nn.Module):
    """RadixMLP-enabled attention layer with variable length support."""

    def __init__(self, config: RadixMLPQwen3Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_flash_attn_varlen = True  # Always true

        # Standard attention projections
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # Normalization layers
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seq_lengths: torch.Tensor,
        max_seq_len: int,
        fold_gather: Optional[torch.Tensor] = None,
        scatter_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with variable length and radix support.

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]
            cos: Cosine values for rotary embeddings [num_tokens, head_dim//2]
            sin: Sine values for rotary embeddings [num_tokens, head_dim//2]
            cu_seq_lengths: Cumulative sequence lengths [batch_size + 1]
            max_seq_len: Maximum sequence length in batch
            fold_gather: Optional indices for radix folding
            scatter_indices: Optional indices for radix scattering

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.shape[0]

        # Choose between radix and standard varlen attention
        if fold_gather is not None and scatter_indices is not None:
            return self._forward_varlen_radix(
                hidden_states,
                cos,
                sin,
                cu_seq_lengths,
                max_seq_len,
                fold_gather,
                scatter_indices,
            )
        else:
            return self._forward_varlen(hidden_states, cos, sin, cu_seq_lengths, max_seq_len)

    def _forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seq_lengths: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Variable length attention forward pass using flash_attn_varlen_func."""
        num_tokens = hidden_states.shape[0]

        # Project to Q, K, V
        q = self.q_proj(hidden_states)  # [num_tokens, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [num_tokens, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)  # [num_tokens, num_kv_heads * head_dim]

        # Reshape and normalize
        q = q.view(num_tokens, self.config.num_attention_heads, self.head_dim)
        k = k.view(num_tokens, self.config.num_key_value_heads, self.head_dim)
        v = v.view(num_tokens, self.config.num_key_value_heads, self.head_dim)

        q = self.q_norm(q.transpose(0, 1)).transpose(0, 1)  # [num_tokens, num_heads, head_dim]
        k = self.k_norm(k.transpose(0, 1)).transpose(0, 1)  # [num_tokens, num_kv_heads, head_dim]

        # Apply rotary embeddings
        q = apply_rotary_pos_emb_single(q, cos, sin)
        k = apply_rotary_pos_emb_single(k, cos, sin)

        # Use flash attention varlen
        if flash_attn_varlen_func is not None:
            attn_output = flash_attn_varlen_func(
                q.unsqueeze(0),  # [1, num_tokens, num_heads, head_dim]
                k.unsqueeze(0),  # [1, num_tokens, num_kv_heads, head_dim]
                v.unsqueeze(0),  # [1, num_tokens, num_kv_heads, head_dim]
                cu_seqlens_q=cu_seq_lengths,
                cu_seqlens_k=cu_seq_lengths,
                max_seqlen_q=max_seq_len,
                max_seqlen_k=max_seq_len,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                softmax_scale=self.scaling,
                causal=self.is_causal,
            )
            attn_output = attn_output.squeeze(0)  # [num_tokens, num_heads, head_dim]
        else:
            raise RuntimeError("flash_attn_varlen_func is not available")

        # Reshape and apply output projection
        attn_output = attn_output.view(num_tokens, -1)  # [num_tokens, hidden_size]
        attn_output = self.o_proj(attn_output)

        return attn_output

    def _forward_varlen_radix(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seq_lengths: torch.Tensor,
        max_seq_len: int,
        fold_gather: torch.Tensor,
        scatter_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Variable length attention with radix folding/scattering."""
        # 1. Fold: apply radix folding to reduce computation following Rust ground truth
        # compact[j] = original[fold_gather[j]]
        hidden_states_compact = torch.index_select(
            hidden_states, dim=0, index=fold_gather
        )  # [compact_tokens, hidden_size]
        cos_compact = torch.index_select(
            cos, dim=0, index=fold_gather
        )  # [compact_tokens, head_dim//2]
        sin_compact = torch.index_select(
            sin, dim=0, index=fold_gather
        )  # [compact_tokens, head_dim//2]

        compact_num_tokens = hidden_states_compact.shape[0]

        # 2. Compute attention in compact space
        # Project to Q, K, V
        q_compact = self.q_proj(hidden_states_compact)  # [compact_tokens, num_heads * head_dim]
        k_compact = self.k_proj(hidden_states_compact)  # [compact_tokens, num_kv_heads * head_dim]
        v_compact = self.v_proj(hidden_states_compact)  # [compact_tokens, num_kv_heads * head_dim]

        # Reshape and normalize
        q_compact = q_compact.view(
            compact_num_tokens, self.config.num_attention_heads, self.head_dim
        )
        k_compact = k_compact.view(
            compact_num_tokens, self.config.num_key_value_heads, self.head_dim
        )
        v_compact = v_compact.view(
            compact_num_tokens, self.config.num_key_value_heads, self.head_dim
        )

        q_compact = self.q_norm(q_compact.transpose(0, 1)).transpose(0, 1)
        k_compact = self.k_norm(k_compact.transpose(0, 1)).transpose(0, 1)

        # Apply rotary embeddings
        q_compact = apply_rotary_pos_emb_single(q_compact, cos_compact, sin_compact)
        k_compact = apply_rotary_pos_emb_single(k_compact, cos_compact, sin_compact)

        # Use flash attention for compact sequences
        if flash_attn_varlen_func is not None:
            # For radix, we need to adjust cu_seq_lengths for compact space
            # This is complex - for now, treat as single sequence
            cu_seqlens_compact = torch.tensor(
                [0, compact_num_tokens], dtype=torch.int32, device=hidden_states.device
            )

            attn_output_compact = flash_attn_varlen_func(
                q_compact.unsqueeze(0),  # [1, compact_tokens, num_heads, head_dim]
                k_compact.unsqueeze(0),  # [1, compact_tokens, num_kv_heads, head_dim]
                v_compact.unsqueeze(0),  # [1, compact_tokens, num_kv_heads, head_dim]
                cu_seqlens_q=cu_seqlens_compact,
                cu_seqlens_k=cu_seqlens_compact,
                max_seqlen_q=compact_num_tokens,
                max_seqlen_k=compact_num_tokens,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                softmax_scale=self.scaling,
                causal=self.is_causal,
            )
            attn_output_compact = attn_output_compact.squeeze(
                0
            )  # [compact_tokens, num_heads, head_dim]
        else:
            raise RuntimeError("flash_attn_varlen_func is not available")

        # 3. Scatter: expand back to original space following Rust ground truth
        # unfolded[i] = compact[scatter_indices[i]]
        attn_output_compact = attn_output_compact.view(
            compact_num_tokens, -1
        )  # [compact_tokens, hidden_size]
        attn_output_flat = torch.index_select(
            attn_output_compact, dim=0, index=scatter_indices
        )  # [original_tokens, hidden_size]

        # Apply output projection
        attn_output = self.o_proj(attn_output_flat)

        return attn_output


class RadixMLPQwen3DecoderLayer(nn.Module):
    """RadixMLP-enabled decoder layer."""

    def __init__(self, config: RadixMLPQwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_radix_mlp = config.use_radix_mlp

        self.self_attn = RadixMLPQwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = RadixMLPQwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = (
            config.layer_types[layer_idx] if hasattr(config, "layer_types") else "full_attention"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seq_lengths: torch.Tensor,
        max_seq_len: int,
        fold_gather: Optional[torch.Tensor] = None,
        scatter_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with radix and varlen support."""
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            cu_seq_lengths=cu_seq_lengths,
            max_seq_len=max_seq_len,
            fold_gather=fold_gather,
            scatter_indices=scatter_indices,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            fold_gather=fold_gather,
            scatter_indices=scatter_indices,
        )
        hidden_states = residual + hidden_states
        return hidden_states


class RadixMLPQwen3Model(nn.Module):
    """RadixMLP-enabled Qwen3 model with variable length support."""

    config_class = RadixMLPQwen3Config

    def __init__(self, config: RadixMLPQwen3Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_radix_mlp = config.use_radix_mlp

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                RadixMLPQwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = (
            hasattr(config, "layer_types") and "sliding_attention" in config.layer_types
        )

    def _prepare_radix_indices(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seq_lengths: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepare radix folding/scattering indices following Rust ground truth.

        Args:
            input_ids: [num_tokens] input token IDs
            position_ids: [num_tokens] position IDs
            cu_seq_lengths: [batch_size + 1] cumulative sequence lengths

        Returns:
            Tuple of (fold_gather, scatter_indices)
        """
        if (
            not self.use_radix_mlp
            or not RADIX_MLP_AVAILABLE
            or compute_fold_and_scatter_torch is None
        ):
            return None, None

        try:
            # Compute radix folding/scattering indices using Rust implementation
            compact_input_ids, compact_position_ids, scatter_indices, fold_gather = (
                compute_fold_and_scatter_torch(
                    input_ids,
                    position_ids,
                    cu_seq_lengths,
                    pad_multiple_of=self.config.radix_pad_multiple_of,
                )
            )

            return fold_gather, scatter_indices

        except Exception as e:
            print(f"Warning: Radix processing failed ({e}), using standard approach")
            return None, None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seq_lengths: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """
        Forward pass with radix and varlen support.

        Args:
            input_ids: [num_tokens] input token IDs
            position_ids: [num_tokens] position IDs
            cu_seq_lengths: [batch_size + 1] cumulative sequence lengths
            max_seq_len: Maximum sequence length in batch

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)  # [num_tokens, hidden_size]

        # Generate rotary embeddings
        cos, sin = self.rotary_emb(position_ids)  # [num_tokens, head_dim//2] each

        # Prepare radix indices
        fold_gather, scatter_indices = self._prepare_radix_indices(
            input_ids, position_ids, cu_seq_lengths
        )

        # Forward through layers
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                cos=cos,
                sin=sin,
                cu_seq_lengths=cu_seq_lengths,
                max_seq_len=max_seq_len,
                fold_gather=fold_gather,
                scatter_indices=scatter_indices,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class RadixMLPQwen3ForCausalLM(nn.Module):
    """RadixMLP-enabled Qwen3 model for causal language modeling."""

    config_class = RadixMLPQwen3Config

    def __init__(self, config: RadixMLPQwen3Config):
        super().__init__()
        self.config = config
        self.model = RadixMLPQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seq_lengths: torch.Tensor,
        max_seq_len: int,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        Forward pass for causal language modeling.

        Args:
            input_ids: [num_tokens] input token IDs
            position_ids: [num_tokens] position IDs
            cu_seq_lengths: [batch_size + 1] cumulative sequence lengths
            max_seq_len: Maximum sequence length in batch
            labels: Optional [num_tokens] labels for loss computation

        Returns:
            Output with loss and logits
        """
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seq_lengths=cu_seq_lengths,
            max_seq_len=max_seq_len,
        )

        # Compute logits
        logits = self.lm_head(hidden_states)  # [num_tokens, vocab_size]

        loss = None
        if labels is not None:
            # Simple cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        # Return a simple output structure
        class Output:
            def __init__(self, loss=None, logits=None):
                self.loss = loss
                self.logits = logits

        return Output(
            loss=loss,
            logits=logits,
        )


# Export classes
__all__ = [
    "RadixMLPQwen3Config",
    "RadixMLPQwen3MLP",
    "RadixMLPQwen3Attention",
    "RadixMLPQwen3DecoderLayer",
    "RadixMLPQwen3Model",
    "RadixMLPQwen3ForCausalLM",
]
