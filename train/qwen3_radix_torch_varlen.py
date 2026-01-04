"""
RadixMLP-enabled Qwen3ForCausalLM with variable length support.

This module implements a Qwen3 model with RadixMLP prefix-based computation sharing
and variable length sequence support using flash attention.

Design principles based on Rust ground truth:
- Batchless architecture: All tensors are num_tokens long (no batch dimension)
- RadixMLP folding/scattering for prefix sharing following Rust implementation
- Variable length sequence support with cu_seq_lengths
- Uses torch.index_select for efficient fold/scatter operations
- needs to run on CUDA. CPU is not supported due to varlen dependency. and fp16/bf16 only.
"""

from typing import Optional, Tuple, List, Any

import torch
from torch import nn
from torch.nn import functional as F

# both RadixMLP and flash attention are REQUIRED dependencies


# required deps
from radix_mlp.torch import compute_fold_and_scatter_torch
from flash_attn import flash_attn_varlen_func


def _get_attn_implementation_config(attn_implementation: str) -> Tuple[bool, bool]:
    """
    Get attention implementation configuration from attn_implementation string.

    Args:
        attn_implementation: String specifying attention implementation

    Returns:
        Tuple of (use_flash_attn, force_fp32_sdpa)
    """
    if attn_implementation == "flash_attention_2":
        return True, False
    elif attn_implementation == "sdpa":
        return False, True
    elif attn_implementation == "sdpa_fp16":
        return False, False
    elif attn_implementation == "eager":
        return False, True  # Use SDPA fallback for eager
    else:
        # Default to flash attention
        return True, False


def flash_attn_varlen_func_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float = 1.0,
    causal: bool = True,
    use_flash_attn: bool = True,
    force_fp32: bool = False,
) -> torch.Tensor:
    """
    Interface for flash_attn_varlen_func with SDPA fallback option.

    Args:
        q: Query tensor [total_tokens, num_heads, head_dim]
        k: Key tensor [total_tokens, num_kv_heads, head_dim]
        v: Value tensor [total_tokens, num_kv_heads, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for queries [batch_size + 1]
        cu_seqlens_k: Cumulative sequence lengths for keys/values [batch_size + 1]
        max_seqlen_q: Maximum sequence length for queries
        max_seqlen_k: Maximum sequence length for keys/values
        dropout_p: Dropout probability
        softmax_scale: Softmax scaling factor
        causal: Whether to apply causal masking
        use_flash_attn: Whether to use flash_attn_varlen_func (True) or SDPA fallback (False)
        force_fp32: Whether to force fp32 computation (only used with SDPA fallback)

    Returns:
        Attention output tensor [total_tokens, num_heads, head_dim]
    """
    if use_flash_attn:
        # Use original flash_attn_varlen_func
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    else:
        # SDPA fallback with for-loop over sequence lengths
        return _sdpa_varlen_fallback(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            force_fp32=force_fp32,
        )


def _sdpa_varlen_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    force_fp32: bool,
) -> torch.Tensor:
    """
    SDPA fallback implementation using for-loop over sequence lengths.

    This function processes each sequence in the batch individually using standard
    PyTorch SDPA, allowing fp32 computation while maintaining compatibility with
    the variable-length interface.

    Note: SDPA always applies dropout according to dropout_p. To disable dropout
    during evaluation, ensure dropout_p=0.0 is passed when not in training mode.

    GQA (Grouped Query Attention) is handled natively by SDPA using enable_gqa=True
    when num_kv_heads != num_heads.
    """
    original_dtype = q.dtype
    device = q.device

    # Convert to fp32 if requested
    if force_fp32:
        q = q.float()
        k = k.float()
        v = v.float()

    batch_size = len(cu_seqlens_q) - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k.shape[1]

    # Handle GQA manually by repeating KV heads (more reliable than enable_gqa)
    if num_kv_heads != num_heads:
        enable_gqa = True  # GQA already handled manually
    else:
        enable_gqa = False

    outputs = []

    # Process each sequence individually
    for batch_idx in range(batch_size):
        start_q = cu_seqlens_q[batch_idx].item()
        end_q = cu_seqlens_q[batch_idx + 1].item()
        start_k = cu_seqlens_k[batch_idx].item()
        end_k = cu_seqlens_k[batch_idx + 1].item()

        seq_len_q = end_q - start_q
        seq_len_k = end_k - start_k

        if seq_len_q == 0 or seq_len_k == 0:
            # Empty sequence, create zero output
            seq_output = torch.zeros(
                int(seq_len_q), num_heads, head_dim, device=device, dtype=q.dtype
            )
            outputs.append(seq_output)
            continue

        # Extract sequence tensors
        q_seq = q[start_q:end_q].unsqueeze(0)  # [1, seq_len_q, num_heads, head_dim]
        k_seq = k[start_k:end_k].unsqueeze(0)  # [1, seq_len_k, num_heads, head_dim]
        v_seq = v[start_k:end_k].unsqueeze(0)  # [1, seq_len_k, num_heads, head_dim]

        # Transpose for SDPA: [batch, heads, seq_len, head_dim]
        q_seq = q_seq.transpose(1, 2)
        k_seq = k_seq.transpose(1, 2)
        v_seq = v_seq.transpose(1, 2)

        # Apply SDPA - no attn_mask needed since we handle causal with is_causal
        with torch.cuda.amp.autocast(enabled=False):  # Keep in specified precision
            seq_output = F.scaled_dot_product_attention(
                q_seq,
                k_seq,
                v_seq,
                attn_mask=None,  # Not needed - causal handled by is_causal
                dropout_p=dropout_p,
                scale=softmax_scale,
                is_causal=causal,
                enable_gqa=enable_gqa,
            )

        # Transpose back: [batch, seq_len, heads, head_dim] -> [seq_len, heads, head_dim]
        seq_output = seq_output.transpose(1, 2).squeeze(0)
        outputs.append(seq_output)

    # Concatenate all sequence outputs
    result = torch.cat(outputs, dim=0)

    # Convert back to original dtype
    if force_fp32:
        result = result.to(original_dtype)

    return result


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
        head_dim: Optional[int] = None,
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

        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )

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


def index_select_scatter_gather(
    input_t: torch.Tensor, indices: torch.Tensor, impl=0
) -> torch.Tensor:
    """Helper function to index select using scatter/gather indices."""
    if impl == 0:
        return input_t.index_select(0, indices)
    elif impl == 1:
        dtype, device = input_t.dtype, input_t.device
        input_t = input_t.contiguous().cpu()
        indices = indices.contiguous().cpu()
        result = torch.index_select(input_t, dim=0, index=indices)
        return result.to(dtype=dtype, device=device)
    else:
        return IndexSelectBackward.apply(input_t, indices)


class IndexSelectBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_t: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices, torch.tensor(input_t.size(0)))
        return input_t.index_select(0, indices)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        indices, original_size = ctx.saved_tensors
        grad_input = torch.zeros(
            original_size.item(),
            *grad_output.shape[1:],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.index_add_(0, indices, grad_output)
        return grad_input, None


def apply_rotary_pos_emb_single(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embeddings to a single tensor."""
    # x: [num_tokens, num_heads, head_dim]
    # cos, sin: [num_tokens, head_dim//2]

    head_dim = x.shape[-1]
    rot_dim = cos.shape[-1] * 2

    # Split x into two halves
    x1 = x[..., : rot_dim // 2]
    x2 = x[..., rot_dim // 2 : rot_dim]

    # Apply rotation using standard formula
    cos_expanded = cos.unsqueeze(-2)  # [num_tokens, 1, head_dim//2]
    sin_expanded = sin.unsqueeze(-2)  # [num_tokens, 1, head_dim//2]

    x1_rot = x1 * cos_expanded - x2 * sin_expanded
    x2_rot = x1 * sin_expanded + x2 * cos_expanded

    x_rotated = torch.cat([x1_rot, x2_rot], dim=-1)

    # If rot_dim < head_dim, keep the remaining dimensions unchanged
    if rot_dim < head_dim:
        x_rotated = torch.cat([x_rotated, x[..., rot_dim:]], dim=-1)

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
        dim = config.head_dim // 2  # Only need half the dimensions for cos/sin
        attention_factor = 1.0

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
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
        inv_freq_expanded = self.inv_freq.float().to(position_ids.device)
        position_ids_expanded = position_ids.float().unsqueeze(-1)

        with torch.cuda.amp.autocast(enabled=False):  # Force float32
            freqs = (
                position_ids_expanded.float() @ inv_freq_expanded.float().unsqueeze(0)
            ).squeeze(-1)  # [num_tokens, head_dim//2]
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
        **kwargs,
    ):
        super().__init__(**kwargs)


class RadixMLPQwen3MLP(nn.Module):
    """RadixMLP-enabled MLP layer with folding/scattering support."""

    def __init__(self, config: RadixMLPQwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Rust-style MLP: separate projections but concatenated during forward
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with optional radix folding/scattering.

        Args:
            x: Input tensor [num_tokens, hidden_size] (already in correct space)

        Returns:
            Output tensor with same shape as input
        """
        # Following Rust implementation: concatenate gate/up during forward
        gate_states = self.gate_proj(x)  # [num_tokens, intermediate_size]
        up_states = self.up_proj(x)  # [num_tokens, intermediate_size]
        down_proj = self.down_proj(self.act_fn(gate_states) * up_states)
        return down_proj


class RadixMLPQwen3Attention(nn.Module):
    """RadixMLP-enabled attention layer with variable length support."""

    def __init__(self, config: RadixMLPQwen3Config, layer_idx: int):
        super().__init__()
        self.layer_type = (
            config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        )
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Standard attention projections - match actual loaded model
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,  # Match actual loaded model
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
        fold_gather: torch.Tensor,
        scatter_indices: torch.Tensor,
        attn_implementation: str = "flash_attention_2",
    ) -> torch.Tensor:
        """Variable length attention with radix folding/scattering."""
        compact_num_tokens = hidden_states.shape[0]

        q_compact = self.q_proj(hidden_states)  # [compact_tokens, num_heads * head_dim]
        k_compact = self.k_proj(
            hidden_states
        )  # [compact_tokens, num_kv_heads * head_dim]
        v_compact = self.v_proj(
            hidden_states
        )  # [compact_tokens, num_kv_heads * head_dim]

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
        q_compact = apply_rotary_pos_emb_single(
            q_compact, cos, sin
        )  # cos and sin are [compact_tokens, head_dim//2]
        k_compact = apply_rotary_pos_emb_single(k_compact, cos, sin)

        # Following Rust implementation: scatter to ORIGINAL space for attention
        skip_radix = (
            scatter_indices.shape[0] == fold_gather.shape[0]
            and (scatter_indices == fold_gather).all().item()
        )
        if skip_radix:
            q = q_compact
            k = k_compact
            v = v_compact
        else:
            q = index_select_scatter_gather(
                q_compact, scatter_indices
            )  # [original_tokens, num_heads, head_dim]b
            k = index_select_scatter_gather(
                k_compact, scatter_indices
            )  # [original_tokens, num_kv_heads, head_dim]
            v = index_select_scatter_gather(
                v_compact, scatter_indices
            )  # [original_tokens, num_kv_heads, head_dim]
        if 1:
            # Determine attention implementation from forward pass parameter
            use_flash_attn, force_fp32 = _get_attn_implementation_config(attn_implementation)

            q_dtype = q.dtype
            # Only force fp16 when using flash attention (it requires fp16)
            if use_flash_attn and q_dtype != torch.float16 and q_dtype != torch.bfloat16:
                q = q.to(torch.float16)
                k = k.to(torch.float16)
                v = v.to(torch.float16)

            # Ensure cu_seq_lengths has correct dtype for flash attention
            cu_seq_lengths_for_attn = (
                cu_seq_lengths.to(torch.int32) if use_flash_attn else cu_seq_lengths
            )

            attn_output = flash_attn_varlen_func_interface(
                q.contiguous(),  # [original_tokens, num_heads, head_dim]
                k.contiguous(),  # [original_tokens, num_kv_heads, head_dim]
                v.contiguous(),  # [original_tokens, num_kv_heads, head_dim]
                cu_seqlens_q=cu_seq_lengths_for_attn,
                cu_seqlens_k=cu_seq_lengths_for_attn,
                max_seqlen_q=max_seq_len,
                max_seqlen_k=max_seq_len,
                dropout_p=0.0,  # qwen3 uses 0.0 dropout for training and eval.
                softmax_scale=self.scaling,
                causal=self.is_causal,
                use_flash_attn=use_flash_attn,
                force_fp32=force_fp32,
            )
            if attn_output.dtype != q_dtype:
                attn_output = attn_output.to(q_dtype)

            # Following Rust: fold back to COMPACT space before o_proj
            attn_output = attn_output.view(
                -1, self.config.num_attention_heads * self.head_dim
            )  # [original_tokens, hidden_size]
        if skip_radix:
            attn_output_compact = attn_output
        else:
            attn_output_compact = index_select_scatter_gather(
                attn_output, fold_gather
            )  # [compact_tokens, hidden_size]
        # Apply output projection
        attn_output_compact = self.o_proj(attn_output_compact)

        # Stay in compact space - MLP runs in compact space
        return attn_output_compact


class RadixMLPQwen3DecoderLayer(nn.Module):
    """RadixMLP-enabled decoder layer."""

    def __init__(self, config: RadixMLPQwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = RadixMLPQwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = RadixMLPQwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = (
            config.layer_types[layer_idx]
            if hasattr(config, "layer_types")
            else "full_attention"
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
        attn_implementation: str = "flash_attention_2",
    ) -> torch.Tensor:
        """Forward pass with radix and varlen support."""
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            cu_seq_lengths=cu_seq_lengths.to(torch.int32).contiguous(),
            max_seq_len=max_seq_len,
            fold_gather=fold_gather,
            scatter_indices=scatter_indices,
            attn_implementation=attn_implementation,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
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

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
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

    @staticmethod
    @torch.no_grad()
    def _prepare_radix_indices(
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seq_lengths: torch.Tensor,
        use_radix_mlp: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare radix folding/scattering indices following Rust ground truth.

        Args:
            input_ids: [num_tokens] input token IDs
            position_ids: [num_tokens] position IDs
            cu_seq_lengths: [batch_size + 1] cumulative sequence lengths

        Returns:
            Tuple of (fold_gather, scatter_indices)
        """
        if not use_radix_mlp:
            identity = torch.arange(input_ids.shape[0]).to(input_ids.device)
            return identity, identity

        # Compute radix folding/scattering indices using Rust implementation
        compact_input_ids, compact_position_ids, scatter_indices, fold_gather = (
            compute_fold_and_scatter_torch(
                input_ids,
                position_ids,
                cu_seq_lengths,
            )
        )

        return fold_gather.to(input_ids.device), scatter_indices.to(input_ids.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seq_lengths: torch.Tensor,
        max_seq_len: int,
        use_radix_mlp: bool = True,
        attn_implementation: str = "flash_attention_2",
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
        fold_gather, scatter_indices = self._prepare_radix_indices(
            input_ids, position_ids, cu_seq_lengths, use_radix_mlp=use_radix_mlp
        )
        skip_radix = (
            scatter_indices.shape[0] == fold_gather.shape[0]
            and (scatter_indices == fold_gather).all().item()
        )

        if skip_radix:
            input_ids_compact = input_ids
            position_ids_compact = position_ids
        else:
            input_ids_compact = index_select_scatter_gather(input_ids, fold_gather)
            position_ids_compact = index_select_scatter_gather(
                position_ids, fold_gather
            )
        # Embed tokens
        hidden_states = self.embed_tokens(
            input_ids_compact
        )  # [num_tokens, hidden_size]

        # Generate rotary embeddings
        cos, sin = self.rotary_emb(
            position_ids_compact
        )  # [num_tokens, head_dim//2] each

        # Prepare radix indices

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
                attn_implementation=attn_implementation,
            )

        hidden_states = self.norm(hidden_states)

        # Following Rust: scatter final outputs back to original layout
        if use_radix_mlp and not skip_radix:
            hidden_states = index_select_scatter_gather(hidden_states, scatter_indices)

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
        use_radix_mlp: bool = True,
        attn_implementation: str = "flash_attention_2",
    ) -> Any:
        """
        Forward pass for causal language modeling.

        Args:
            input_ids: [num_tokens] input token IDs
            position_ids: [num_tokens] position IDs
            cu_seq_lengths: [batch_size + 1] cumulative sequence lengths
            max_seq_len: Maximum sequence length in batch
            labels: Optional [num_tokens] labels for loss computation
            use_radix_mlp: Whether to use RadixMLP folding/scattering

        Returns:
            Output with loss and logits
        """
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seq_lengths=cu_seq_lengths,
            max_seq_len=max_seq_len,
            use_radix_mlp=use_radix_mlp,
            attn_implementation=attn_implementation,
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
