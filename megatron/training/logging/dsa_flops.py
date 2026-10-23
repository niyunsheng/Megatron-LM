# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSA attention and lightning-indexer FLOPs accounting."""

import math
from argparse import Namespace

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, parse_hybrid_pattern
from megatron.core.transformer.experimental_attention_variant.dsa import is_dsa_skip_topk_layer


def _dsa_sparse_core_scale(total_real_tokens, seqlen_squared_sum, dsa_indexer_topk):
    """Fraction of dense causal (query, key) pairs that DSA actually attends to.

    A DSA layer scores every past position with the indexer but runs attention
    only over the ``dsa_indexer_topk`` highest-scoring keys, so its core
    attention cost is ``sum_i(min(i, topk))`` pairs per sequence instead of the
    dense causal ``L^2 / 2``. Returns the ratio between the two, i.e. the
    factor the dense core-attention coefficient must be scaled by.

    The caller only has the batch aggregates ``sum_i(L_i)`` and
    ``sum_i(L_i ** 2)``, not the individual sequence lengths, so the ratio is
    evaluated at the length-weighted mean ``sum(L^2) / sum(L)``. For equal-length
    sequences, this aggregation introduces no additional approximation, but
    the pair counts still use the continuous approximation described below.
    For ragged batches, it weights toward the long sequences that dominate
    attention cost. Collapses to ``1.0`` when the sequences are
    no longer than ``topk``, where top-k selects everything and attention is
    dense.
    """
    if not dsa_indexer_topk or total_real_tokens <= 0 or seqlen_squared_sum <= 0:
        return 1.0
    mean_seqlen = seqlen_squared_sum / total_real_tokens
    # Average attended KV entries per query: ``min(i, topk)`` averaged over the
    # sequence, approximated as ``eff * (1 - eff / (2 * L))`` with
    # ``eff = min(topk, floor(L))``. The exact discrete average has ``eff - 1``
    # in the correction term; the O(1/L) difference is below the precision of
    # this estimate.
    eff = min(dsa_indexer_topk, math.floor(mean_seqlen))
    attended = eff * (1 - eff / (2 * mean_seqlen))
    # Dense causal attention averages ~``mean_seqlen / 2`` keys per query.
    return attended / (mean_seqlen / 2)


def _dsa_indexer_flops(
    *,
    hidden_size,
    q_lora_rank,
    n_heads,
    head_dim,
    num_indexer_layers,
    dsa_indexer_loss_enabled: bool,
    dsa_indexer_use_sparse_loss=False,
    sparse_core_scale=1.0,
):
    """DSA lightning-indexer FLOPs coefficients, fwd/bwd expansion included.

    Counts the indexer's projections (a Q projection off the shared ``q_lora``
    residual, the ``linear_wk`` key path, and the per-head ``weights_proj``)
    plus its dense scoring pass of every query against every past token under
    a causal mask. Scoring is ``O(L^2)`` even though the attention consuming
    it is sparse. The indexer KL-loss computation and top-k selection are
    excluded from this estimate.

    Only ``num_indexer_layers`` layers pay: with cross-layer index sharing
    (``dsa_indexer_topk_freq``) the layers in between reuse the most recent
    top-k (see ``is_dsa_skip_topk_layer``).

    The indexer does NOT get the global fwd+bwd factor of 3. It is trained
    only by its own KL loss, so ``DSAttention.forward`` runs it under
    ``torch.no_grad()`` unless ``dsa_indexer_loss_coeff > 0`` (which defaults
    to None), and even then ``x`` / ``qr`` are detached so no gradient leaves
    the indexer:

      * loss off -> forward only, everything is 1x.
      * loss on  -> the projections (wq_b / wk / weights_proj) read a detached
        input, so autograd skips their dgrad and they pay fwd + wgrad = 2x.
        The scoring GEMM has two activation operands that both need gradients
        to reach those weights, so it pays fwd + dq + dk. Forward scoring is
        always dense (top-k selection needs every score). How much of the
        backward is dense depends on the loss variant:

          - dense KL (``dsa_indexer_use_sparse_loss=False``, the default):
            every causal (query, key) pair carries a gradient, so
            dq + dk are dense too and the scoring pays 3x.
          - sparse KL (``dsa_indexer_use_sparse_loss=True``): the KL is
            taken over the selected top-k keys only, so the score gradient is
            zero outside them and dq + dk span just the top-k pairs. The
            scoring pays ``1 + 2 * sparse_core_scale``, with
            ``sparse_core_scale`` the same top-k / dense pair ratio that
            ``_dsa_sparse_core_scale`` applies to core attention. The fused
            cuDNN backend (``_compute_sparse_indexer_loss_and_grads``)
            executes exactly this sparse backward; the reference PyTorch path
            multiplies a dense zero-padded gradient instead, which is an
            implementation detail and, like the dense-masked reference
            attention, does not enter the model FLOPs count.

    Whether the indexer is trained is part of the training procedure rather
    than the kernel schedule, so it belongs in this model-FLOPs count.
    The caller derives ``dsa_indexer_loss_enabled`` from the configured loss
    coefficient; its magnitude affects gradients but not the FLOPs count.

    Returns ``(token_linear, core)`` INCLUDING the fwd/bwd and FMA factors.
    Multiply ``token_linear`` by the real token count and ``core`` by
    ``sum_i(L_i ** 2)``.
    """
    if num_indexer_layers <= 0:
        return 0, 0
    if q_lora_rank is None:
        # Mirrors DSAIndexer's own fallback when the model has no q lora rank.
        q_lora_rank = hidden_size
    index_dim = n_heads * head_dim
    token_linear = num_indexer_layers * (
        q_lora_rank * index_dim  # wq_b
        + hidden_size * head_dim  # wk
        + hidden_size * n_heads  # weights_proj
    )
    # Scoring each query against every past token under a causal mask (/2).
    core = num_indexer_layers * index_dim / 2
    fma_expansion_factor = 2
    if not dsa_indexer_loss_enabled:
        token_expansion, core_expansion = 1, 1
    else:
        # Projections: fwd + wgrad (input is detached, no dgrad).
        token_expansion = 2
        # Scoring: dense fwd + dq + dk, where dq/dk cover only the top-k pairs
        # under the sparse KL loss (see docstring).
        core_expansion = 1 + 2 * (sparse_core_scale if dsa_indexer_use_sparse_loss else 1.0)
    return (
        token_expansion * fma_expansion_factor * token_linear,
        core_expansion * fma_expansion_factor * core,
    )


def _num_dsa_indexer_layers(num_layers, skip_topk_offset, topk_freq):
    """Count layers that compute their own DSA index (the rest reuse one).

    On the standard-model path every layer is a DSA attention layer (MTP
    layers included -- ``DSAttention.__init__`` numbers them
    ``layer_number + config.num_layers``, which is exactly how the caller
    extends ``num_layers``), so the predicate runs over the whole
    ``1..num_layers`` range.
    """
    return sum(
        1
        for layer_number in range(1, num_layers + 1)
        if not is_dsa_skip_topk_layer(layer_number, skip_topk_offset or 0, topk_freq or 1)
    )


def _num_dsa_indexer_layers_in_pattern(layer_types, dsa_symbol, skip_topk_offset, topk_freq):
    """Count 'D' positions in a hybrid layer-type sequence that compute their
    own DSA index.

    ``layer_types`` is one hybrid block's layer sequence (main pattern with
    pipes stripped, or one MTP depth's pattern). ``DSAttention`` receives the
    block-local 1-indexed layer number (``HybridBlock`` numbers layers
    ``i + 1 + pp_layer_offset`` across pipeline stages; hybrid MTP blocks
    restart at 1 per depth with ``pp_layer_offset=0``), and the skip predicate
    runs on that number -- non-DSA positions advance the numbering but never
    compute an index.
    """
    return sum(
        1
        for layer_number, symbol in enumerate(layer_types, start=1)
        if symbol == dsa_symbol
        and not is_dsa_skip_topk_layer(layer_number, skip_topk_offset or 0, topk_freq or 1)
    )


def _dsa_layer_flops(args, total_real_tokens, seqlen_squared_sum):
    """Per-DSA-layer FLOPs coefficients for the hybrid-model path.

    Returns ``(token_linear, core)`` INCLUDING the fwd+bwd (x3) and FMA (x2)
    expansion factors. Multiply ``token_linear`` by the real token count and
    ``core`` by ``sum_i(L_i ** 2)``. The formulas mirror the standard-path
    ``"dsa"`` branch in ``transformer_flops``:

      * token-linear: the plain-MLA projection cost (q/kv lora, RoPE, output
        projection). Absorption relocates the same ``W_UK``/``W_UV`` GEMMs, so
        the absorbed form costs the same per token.
      * core: the absorbed-MLA form (``QK^T`` over
        ``kv_lora_rank + qk_pos_emb_head_dim`` per head, ``AV`` over
        ``kv_lora_rank``), scaled down to the indexer's top-k pair count.

    The indexer is NOT included here -- only ``dsa_indexer_topk_freq``-spaced
    layers pay it; callers add ``_dsa_indexer_flops`` separately.
    """
    if args.q_lora_rank is None:
        q_term = (
            args.hidden_size
            * args.num_attention_heads
            * (args.qk_head_dim + args.qk_pos_emb_head_dim)
        )
    else:
        q_term = args.q_lora_rank * (
            args.hidden_size
            + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
            + 1
        )
    forward_backward_expansion_factor = 3
    fma_expansion_factor = 2
    token_linear = (
        forward_backward_expansion_factor
        * fma_expansion_factor
        * (
            ## q lora + rope + q norm
            q_term
            ## kv lora + rope + kv norm
            + args.kv_lora_rank
            * (
                args.hidden_size
                + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
                + 1
            )
            + args.hidden_size * args.qk_pos_emb_head_dim
            ## o proj
            + (args.num_attention_heads * args.v_head_dim) * args.hidden_size
        )
    )
    core = (
        forward_backward_expansion_factor
        * fma_expansion_factor
        * (
            args.num_attention_heads * (args.kv_lora_rank + args.qk_pos_emb_head_dim) / 2
            + args.num_attention_heads * args.kv_lora_rank / 2
        )
        * _dsa_sparse_core_scale(total_real_tokens, seqlen_squared_sum, args.dsa_indexer_topk)
    )
    return token_linear, core


def dsa_flops(
    args: Namespace, total_tokens: float, seqlen_squared_sum: float, num_dsa_layers: int
) -> float:
    """Calculate the DSA contribution to hybrid-model training FLOPs.

    Counts absorbed-MLA projections and top-k sparse core attention for every
    DSA layer, plus the lightning indexer on layers that compute their own
    top-k. Both terms already include their forward/backward and FMA factors;
    add this result to ``hybrid_flops`` without another x3 expansion.

    Args:
        args: Megatron arguments, including the hybrid pattern and DSA dimensions.
            The pattern determines DSA layers independently of
            ``experimental_attention_variant``, which hybrid configuration
            sets on the model config rather than on the arguments.
        total_tokens: Real (unpadded) token count in the global batch.
        seqlen_squared_sum: Sum of squared real sequence lengths in that batch.
        num_dsa_layers: DSA layer count from ``get_hybrid_layer_counts``, including
            MTP depths. Zero returns immediately without reading DSA arguments.

    Returns:
        Training FLOPs for DSA attention and its indexer, excluding other hybrid
        layers, logits, and MTP norms/projections accounted for by ``hybrid_flops``.
    """
    if num_dsa_layers == 0:
        return 0.0

    per_layer_token_linear, per_layer_core = _dsa_layer_flops(
        args, total_tokens, seqlen_squared_sum
    )
    # DSAttention's skip predicate runs on block-local layer numbers:
    # the main pattern numbers layers sequentially across pipes, and
    # each hybrid MTP depth restarts at 1 (``pp_layer_offset=0``).
    parsed_pattern = parse_hybrid_pattern(args.hybrid_layer_pattern)
    main_layer_types = (parsed_pattern.main_pattern or "").replace(Symbols.PIPE, "")
    num_indexer_layers = _num_dsa_indexer_layers_in_pattern(
        main_layer_types,
        Symbols.DS_ATTENTION,
        args.dsa_indexer_skip_topk_offset,
        args.dsa_indexer_topk_freq,
    )
    if parsed_pattern.mtp_pattern:
        num_indexer_layers += parsed_pattern.mtp_num_depths * _num_dsa_indexer_layers_in_pattern(
            parsed_pattern.mtp_pattern,
            Symbols.DS_ATTENTION,
            args.dsa_indexer_skip_topk_offset,
            args.dsa_indexer_topk_freq,
        )
    indexer_token_linear, indexer_core = _dsa_indexer_flops(
        hidden_size=args.hidden_size,
        q_lora_rank=args.q_lora_rank,
        n_heads=args.dsa_indexer_n_heads,
        head_dim=args.dsa_indexer_head_dim,
        num_indexer_layers=num_indexer_layers,
        dsa_indexer_loss_enabled=(args.dsa_indexer_loss_coeff or 0.0) > 0,
        dsa_indexer_use_sparse_loss=getattr(args, "dsa_indexer_use_sparse_loss", False),
        sparse_core_scale=_dsa_sparse_core_scale(
            total_tokens, seqlen_squared_sum, args.dsa_indexer_topk
        ),
    )
    dsa_token_linear_term = num_dsa_layers * per_layer_token_linear + indexer_token_linear
    dsa_core_term = num_dsa_layers * per_layer_core + indexer_core
    return total_tokens * dsa_token_linear_term + seqlen_squared_sum * dsa_core_term
