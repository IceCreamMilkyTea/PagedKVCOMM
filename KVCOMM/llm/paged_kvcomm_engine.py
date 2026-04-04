"""Paged KV cache engine for KVCOMM, built on top of nano-vllm's engine.

Reuses nano-vllm's:
  - BlockManager   → block allocation / deallocation / prefix-caching
  - Scheduler      → prefill / decode scheduling
  - ModelRunner     → prepare_prefill / prepare_decode / slot_mapping / block_tables
  - Attention layer → store_kvcache (triton) + flash_attn_with_kvcache (zero-copy)

Adds KVCOMM-specific:
  - Anchor storage using block references (instead of full tensor copies)
  - Block-level KV read/write for delta computation
  - Multi-agent KV reuse through shared block tables

Architecture overview:
  ┌─────────────────────────────────────────────────────────┐
  │                    PagedKVCOMMEngine                    │
  │                                                         │
  │  ┌──────────┐   ┌───────────────┐   ┌───────────────┐   │
  │  │Scheduler │──>│ ModelRunner   │──>│ Model+Attn    │   │
  │  │          │   │ (slot_mapping │   │ (store_kvcache│   │
  │  │ allocate │   │  block_tables)│   │  flash_attn)  │   │
  │  └──────────┘   └───────────────┘   └───────────────┘   │
  │       │                                    │            │
  │       v                                    v            │
  │  ┌──────────┐              ┌────────────────────┐       │
  │  │  Block   │              │  kv_cache tensor   │       │
  │  │  Manager │──────────────│[2,L,num_blk,B,H,D] │       │
  │  │(hash dup)│              └────────────────────┘       │
  │  └──────────┘                      │                    │
  │       │                            v                    │
  │  ┌──────────────────────────────────────────┐           │
  │  │         KVCOMM Anchor Store              │           │
  │  │  anchor = {                              │           │
  │  │    block_table: [b1, b2, b3],  # refs    │           │
  │  │    num_tokens: 48,                       │           │
  │  │    ph_key_emb: tensor,   # for sim       │           │
  │  │    delta_blocks: [d1,d2] # delta KV      │           │
  │  │  }                                       │           │
  │  └──────────────────────────────────────────┘           │
  └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import random
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

# nano-vllm imports
from nanovllm.engine.block_manager import BlockManager, Block
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import store_kvcache


class PagedAnchorEntry:
    """A single anchor stored as block references + lightweight metadata.

    Instead of storing full KV tensor copies (as DynamicCache does),
    we store:
      - block_table: list of block IDs in the kv_cache pool → zero-copy reference
      - num_tokens: how many tokens the anchor covers
      - ph_key_embedding / ph_value_embedding: small tensors for similarity computation
      - per-agent delta block tables: block IDs holding the delta KV
    """

    __slots__ = (
        "block_table",
        "num_tokens",
        "ph_key_embedding",
        "ph_value_embedding",
        "agent_deltas",
    )

    def __init__(
        self,
        block_table: List[int],
        num_tokens: int,
        ph_key_embedding: torch.Tensor,
        ph_value_embedding: torch.Tensor,
    ):
        self.block_table = block_table
        self.num_tokens = num_tokens
        # Keep similarity embeddings on CPU to avoid long-run GPU memory growth.
        self.ph_key_embedding = ph_key_embedding.detach().cpu()
        self.ph_value_embedding = ph_value_embedding.detach().cpu()
        # agent_id → {"ph_delta_blocks": [...], "pf_delta_blocks": [...], ...}
        self.agent_deltas: Dict[str, Dict[str, Any]] = {}


class PagedKVCOMMEngine:
    """Central coordinator for KVCOMM's anchor KV reuse with paged attention.

    This replaces KVCOMMEngine's DynamicCache-based anchor storage with
    block-table-based storage that references the pre-allocated kv_cache pool.

    The kv_cache pool is the same tensor used by nano-vllm's ModelRunner:
        kv_cache shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        kv_cache[0] = all key blocks
        kv_cache[1] = all value blocks

    Reading/writing KV data goes through the same blocks that the model's
    Attention layer writes to during forward pass → true zero-copy.
    """

    def __init__(
        self,
        kv_cache: torch.Tensor,
        block_manager: BlockManager,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        """
        Args:
            kv_cache: Pre-allocated tensor [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
            block_manager: nano-vllm's BlockManager instance
            block_size: tokens per block
            num_layers: number of transformer layers
            num_kv_heads: number of KV attention heads (after TP split)
            head_dim: dimension per head
        """
        self.kv_cache = kv_cache
        self.block_manager = block_manager
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # k_cache[layer] = [num_blocks, block_size, num_kv_heads, head_dim]
        self.k_cache = kv_cache[0]  # [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.v_cache = kv_cache[1]

        # Anchor store: ph_id → message → PagedAnchorEntry
        self.anchors: Dict[str, Dict[str, PagedAnchorEntry]] = {}
        # Anchor activation counts: ph_id → message → activation_count
        # Used by evict_anchor to identify least-used anchors.
        self.anchor_info: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()

        # Response embedding store (separate from placeholder anchors).
        # Stores actual response token value KVs for response anchor prediction.
        # ph_id → message → value tensor [num_layers, num_kv_heads, num_tokens, head_dim] on CPU
        self.response_embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
        # ph_id → message → num_tokens (actual response length)
        self.response_token_counts: Dict[str, Dict[str, int]] = {}

    # ─────────────────────── Block-level KV read/write ───────────────────────

    def read_kv_from_blocks(
        self,
        block_table: List[int],
        num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read KV tensors from blocks. Returns [num_layers, num_kv_heads, num_tokens, head_dim].

        This reconstructs continuous tensors from paged blocks.
        Used for delta computation (set_anchor) and similarity computation (predict_as_anchor).
        """
        num_blocks = len(block_table)
        # Gather blocks: for each layer, collect the relevant blocks
        # k_cache shape: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        block_ids = torch.tensor(block_table, dtype=torch.long, device=self.kv_cache.device)

        # Index into block pool: [num_layers, len(block_table), block_size, num_kv_heads, head_dim]
        k_blocks = self.k_cache[:, block_ids]
        v_blocks = self.v_cache[:, block_ids]

        # Reshape to continuous: [num_layers, total_slots, num_kv_heads, head_dim]
        k_flat = k_blocks.reshape(self.num_layers, -1, self.num_kv_heads, self.head_dim)
        v_flat = v_blocks.reshape(self.num_layers, -1, self.num_kv_heads, self.head_dim)

        # Trim to actual num_tokens (last block may be partially filled)
        k_out = k_flat[:, :num_tokens]  # [num_layers, num_tokens, num_kv_heads, head_dim]
        v_out = v_flat[:, :num_tokens]

        # Transpose to match KVCOMM's convention: [num_layers, num_kv_heads, num_tokens, head_dim]
        k_out = k_out.transpose(1, 2)
        v_out = v_out.transpose(1, 2)

        return k_out, v_out

    def write_kv_to_blocks(
        self,
        block_table: List[int],
        key: torch.Tensor,
        value: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """Write KV tensors into blocks. key/value: [num_layers, num_kv_heads, num_tokens, head_dim].

        Used to store delta KV or modified KV into pre-allocated blocks.
        """
        # Transpose to [num_layers, num_tokens, num_kv_heads, head_dim]
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        if num_tokens <= 0:
            return
        if not block_table:
            raise ValueError("write_kv_to_blocks: empty block_table for positive num_tokens")
        if key.shape[1] != num_tokens or value.shape[1] != num_tokens:
            raise ValueError(
                "write_kv_to_blocks token mismatch: "
                f"num_tokens={num_tokens}, key_tokens={key.shape[1]}, value_tokens={value.shape[1]}"
            )

        block_ids = torch.tensor(block_table, dtype=torch.long, device=self.kv_cache.device)

        for layer_idx in range(self.num_layers):
            # Compute slot_mapping for this write
            slots = []
            tokens_left = num_tokens
            for i, bid in enumerate(block_table):
                n = min(self.block_size, tokens_left)
                for j in range(n):
                    slots.append(bid * self.block_size + j)
                tokens_left -= n

            slot_mapping = torch.tensor(slots[:num_tokens], dtype=torch.int32, device=key.device)

            # Use nano-vllm's triton kernel
            store_kvcache(
                key[layer_idx],      # [num_tokens, num_kv_heads, head_dim]
                value[layer_idx],
                self.k_cache[layer_idx],  # [num_blocks, block_size, num_kv_heads, head_dim]
                self.v_cache[layer_idx],
                slot_mapping,
            )

    def allocate_blocks_for_tokens(self, num_tokens: int) -> List[int]:
        """Allocate fresh blocks from the free pool for a given number of tokens.

        If the free pool is exhausted, attempts to evict least-used anchors
        to reclaim blocks before raising.
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        block_ids = []
        for _ in range(num_blocks_needed):
            if not self.block_manager.free_block_ids:
                # Try to reclaim blocks by evicting least-used anchors
                remaining = num_blocks_needed - len(block_ids)
                if not self.evict_until_free(remaining):
                    # Free any partially allocated blocks before raising
                    self.free_blocks(block_ids)
                    raise RuntimeError(
                        f"Not enough free blocks: need {num_blocks_needed}, "
                        f"have {len(self.block_manager.free_block_ids)} "
                        f"(even after eviction)"
                    )
            bid = self.block_manager.free_block_ids[0]
            block = self.block_manager._allocate_block(bid)
            block_ids.append(bid)
        return block_ids

    def free_blocks(self, block_ids: List[int]) -> None:
        """Return blocks to the free pool."""
        for bid in reversed(block_ids):
            block = self.block_manager.blocks[bid]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.block_manager._deallocate_block(bid)

    def increment_ref(self, block_ids: List[int]) -> None:
        """Increment reference count on blocks (for sharing/forking)."""
        for bid in block_ids:
            self.block_manager.blocks[bid].ref_count += 1

    # ─────────────────────── Anchor Operations ───────────────────────

    def register_base_anchor(
        self,
        ph_id: str,
        message: str,
        block_table: List[int],
        num_tokens: int,
        max_anchor_num: int = 20,
        window_length: int = 5,
    ) -> bool:
        """Register a base anchor entry directly from existing blocks.

        This is used to bootstrap anchor history before dense set_anchor paths
        become available for a placeholder.

        Returns:
            True if a new anchor entry was created, False otherwise.
        """
        if num_tokens <= 0 or not block_table:
            return False

        ph_key, ph_val = self.read_kv_from_blocks(block_table, num_tokens)

        with self._lock:
            ph_store = self.anchors.setdefault(ph_id, {})
            if message in ph_store:
                return False

            # Evict if over capacity
            if len(ph_store) >= max_anchor_num:
                self.evict_anchor(ph_id, window_length)

            self.increment_ref(block_table)
            ph_store[message] = PagedAnchorEntry(
                block_table=list(block_table),
                num_tokens=num_tokens,
                ph_key_embedding=ph_key,
                ph_value_embedding=ph_val,
            )
            self.anchor_info.setdefault(ph_id, {})[message] = 0
            return True

    def set_anchor(
        self,
        agent_id: str,
        ph_id: str,
        message: str,
        real_block_table: List[int],
        real_num_tokens: int,
        base_block_table: List[int],
        base_num_tokens: int,
        real_prefix_block_table: List[int],
        real_prefix_num_tokens: int,
        base_prefix_block_table: List[int],
        base_prefix_num_tokens: int,
        max_anchor_num: int = 20,
        window_length: int = 5,
        free_ratio_threshold: float = 0.15,
    ) -> None:
        """Store an anchor entry using block references + delta.

        All KV reads, delta computation, and block allocation happen BEFORE
        the lock so that evict_until_free (triggered by allocate_blocks_for_tokens
        when the pool is full) cannot see this entry and self-evict it.

        The single final lock atomically: creates or retrieves the entry,
        frees any stale delta blocks from a prior call for this (agent_id,
        message) pair, and stores the new delta block references.
        """
        # ── 0. Proactive eviction (mirrors gpt_chat update_anchor) ───────────
        # Check pool health before allocating delta blocks. Tries this ph_id
        # first, then falls back globally — same order as gpt_chat's per-ph_id
        # update_anchor + global fallback.
        self.evict_proactive(
            ph_id=ph_id,
            free_ratio_threshold=free_ratio_threshold,
            window_length=window_length,
        )

        # ── 1. All KV reads and delta computation (no lock needed) ──────────
        base_ph_key, base_ph_val = self.read_kv_from_blocks(base_block_table, base_num_tokens)

        real_ph_key, real_ph_val = self.read_kv_from_blocks(real_block_table, real_num_tokens)
        if real_ph_key.shape[2] != real_num_tokens or real_ph_key.shape[2] == 0:
            raise ValueError(
                f"set_anchor: real_ph_key has {real_ph_key.shape[2]} tokens at dim 2, "
                f"expected {real_num_tokens}. real_block_table may be empty or misaligned."
            )
        ph_key_delta = real_ph_key - base_ph_key[..., :real_num_tokens, :]
        ph_val_delta = real_ph_val - base_ph_val[..., :real_num_tokens, :]

        # Prefix delta (absent when prefix token counts are zero)
        pf_tokens = 0
        pf_key_delta: Optional[torch.Tensor] = None
        pf_val_delta: Optional[torch.Tensor] = None
        if real_prefix_num_tokens > 0 and base_prefix_num_tokens > 0:
            real_pf_key, real_pf_val = self.read_kv_from_blocks(
                real_prefix_block_table, real_prefix_num_tokens
            )
            base_pf_key, base_pf_val = self.read_kv_from_blocks(
                base_prefix_block_table, base_prefix_num_tokens
            )
            pf_tokens = min(
                real_pf_key.shape[2], base_pf_key.shape[2],
                real_prefix_num_tokens, base_prefix_num_tokens,
            )
            if pf_tokens > 0:
                pf_key_delta = real_pf_key[..., :pf_tokens, :] - base_pf_key[..., :pf_tokens, :]
                pf_val_delta = real_pf_val[..., :pf_tokens, :] - base_pf_val[..., :pf_tokens, :]

        # ── 2. Block allocation and write (no lock, entry not in store yet) ──
        # The entry is absent from ph_store until step 3, so evict_until_free
        # triggered here cannot pick this message → no self-eviction.
        ph_delta_blocks = self.allocate_blocks_for_tokens(real_num_tokens)
        self.write_kv_to_blocks(ph_delta_blocks, ph_key_delta, ph_val_delta, real_num_tokens)

        pf_delta_blocks: List[int] = []
        if pf_tokens > 0 and pf_key_delta is not None:
            pf_delta_blocks = self.allocate_blocks_for_tokens(pf_tokens)
            self.write_kv_to_blocks(pf_delta_blocks, pf_key_delta, pf_val_delta, pf_tokens)

        # ── 3. Single lock: create/update entry + store delta refs atomically ─
        with self._lock:
            ph_store = self.anchors.setdefault(ph_id, {})

            if message not in ph_store:
                if len(ph_store) >= max_anchor_num:
                    self.evict_anchor(ph_id, window_length)
                self.increment_ref(base_block_table)
                entry = PagedAnchorEntry(
                    block_table=list(base_block_table),
                    num_tokens=base_num_tokens,
                    ph_key_embedding=base_ph_key,
                    ph_value_embedding=base_ph_val,
                )
                ph_store[message] = entry
                self.anchor_info.setdefault(ph_id, {})[message] = 0
            else:
                entry = ph_store[message]

            # Free stale delta blocks before overwriting to prevent block leak.
            old_delta = entry.agent_deltas.get(agent_id)
            if old_delta is not None:
                self.free_blocks(old_delta.get("ph_delta_blocks", []))
                self.free_blocks(old_delta.get("pf_delta_blocks", []))

            entry.agent_deltas[agent_id] = {
                "ph_delta_blocks": ph_delta_blocks,
                "ph_delta_num_tokens": real_num_tokens,
                "pf_delta_blocks": pf_delta_blocks,
                "pf_delta_num_tokens": pf_tokens,
            }

    def offset_kv_cache(
        self,
        agent_id: str,
        ph_id: str,
        message: str,
        base_ph_block_table: List[int],
        base_ph_num_tokens: int,
        base_pf_block_table: List[int],
        base_pf_num_tokens: int,
        anchor_list: List[str],
        temperature: float = 1.0,
    ) -> Tuple[List[int], int, List[int], int]:
        """Apply weighted anchor deltas to base KV and write result to new blocks.

        This is the paged equivalent of KVCOMMEngine.offset_kv_cache_pair().

        Returns:
            (new_ph_block_table, ph_num_tokens, new_pf_block_table, pf_num_tokens)
        """
        ph_store = self.anchors.get(ph_id, {})
        if not ph_store or not anchor_list:
            # No anchors, return copies of base blocks
            self.increment_ref(base_ph_block_table)
            self.increment_ref(base_pf_block_table)
            return base_ph_block_table, base_ph_num_tokens, base_pf_block_table, base_pf_num_tokens

        # Collect valid anchors with per-agent deltas that fully cover current spans.
        valid_anchors = []
        for msg in anchor_list:
            entry = ph_store.get(msg)
            if entry is None:
                continue
            delta_info = entry.agent_deltas.get(agent_id)
            if delta_info is None:
                continue
            if entry.num_tokens >= base_ph_num_tokens:
                ph_delta_tokens = int(delta_info.get("ph_delta_num_tokens", 0) or 0)
                if ph_delta_tokens < base_ph_num_tokens:
                    continue
                if base_pf_num_tokens > 0:
                    pf_delta_tokens = int(delta_info.get("pf_delta_num_tokens", 0) or 0)
                    if pf_delta_tokens < base_pf_num_tokens:
                        continue
                valid_anchors.append((msg, entry))

        if not valid_anchors:
            self.increment_ref(base_ph_block_table)
            self.increment_ref(base_pf_block_table)
            return base_ph_block_table, base_ph_num_tokens, base_pf_block_table, base_pf_num_tokens

        # Read base KV
        base_ph_key, base_ph_val = self.read_kv_from_blocks(base_ph_block_table, base_ph_num_tokens)
        base_pf_key, base_pf_val = self.read_kv_from_blocks(base_pf_block_table, base_pf_num_tokens)

        # Compute similarity weights (same logic as KVCOMMEngine) without stacking
        # all anchors into one large tensor (reduces peak VRAM).
        sims_list = []
        for _, entry in valid_anchors:
            anchor_key = entry.ph_key_embedding[..., -base_ph_num_tokens:, :]
            if anchor_key.device != base_ph_key.device:
                anchor_key = anchor_key.to(base_ph_key.device, non_blocking=True)
            sims_list.append((base_ph_key - anchor_key).norm(2, dim=-2))
        sims = torch.stack(sims_list, dim=0)
        weights = torch.softmax(-sims.float() / temperature, dim=0).unsqueeze(-2)

        # Weighted sum of deltas (placeholder)
        ph_delta_sum_k = torch.zeros_like(base_ph_key)
        ph_delta_sum_v = torch.zeros_like(base_ph_val)
        for i, (msg, entry) in enumerate(valid_anchors):
            delta_info = entry.agent_deltas[agent_id]
            dk, dv = self.read_kv_from_blocks(
                delta_info["ph_delta_blocks"],
                base_ph_num_tokens,
            )
            w = weights[i]
            ph_delta_sum_k += w * dk[..., :base_ph_num_tokens, :]
            ph_delta_sum_v += w * dv[..., :base_ph_num_tokens, :]

        # Weighted sum of deltas (prefix)
        pf_delta_sum_k = torch.zeros_like(base_pf_key)
        pf_delta_sum_v = torch.zeros_like(base_pf_val)
        for i, (msg, entry) in enumerate(valid_anchors):
            delta_info = entry.agent_deltas[agent_id]
            if base_pf_num_tokens <= 0:
                continue
            dk, dv = self.read_kv_from_blocks(delta_info["pf_delta_blocks"], base_pf_num_tokens)
            w = weights[i]
            pf_delta_sum_k += w * dk
            pf_delta_sum_v += w * dv

        # Apply delta to base and write to new blocks
        new_ph_key = base_ph_key + ph_delta_sum_k.to(base_ph_key.dtype)
        new_ph_val = base_ph_val + ph_delta_sum_v.to(base_ph_val.dtype)
        new_ph_key[0] = base_ph_key[0]  # Keep layer 0 unchanged (KVCOMM convention)
        new_ph_val[0] = base_ph_val[0]

        new_ph_blocks = self.allocate_blocks_for_tokens(base_ph_num_tokens)
        self.write_kv_to_blocks(new_ph_blocks, new_ph_key, new_ph_val, base_ph_num_tokens)

        new_pf_key = base_pf_key + pf_delta_sum_k.to(base_pf_key.dtype)
        new_pf_val = base_pf_val + pf_delta_sum_v.to(base_pf_val.dtype)
        new_pf_key[0] = base_pf_key[0]
        new_pf_val[0] = base_pf_val[0]

        new_pf_blocks = self.allocate_blocks_for_tokens(base_pf_num_tokens)
        self.write_kv_to_blocks(new_pf_blocks, new_pf_key, new_pf_val, base_pf_num_tokens)

        return new_ph_blocks, base_ph_num_tokens, new_pf_blocks, base_pf_num_tokens

    # ── Local-reference offset (inner-round optimisation) ─────────────

    def offset_kv_cache_local_ref(
        self,
        agent_id: str,
        ph_id: str,
        message: str,
        base_ph_block_table: List[int],
        base_ph_num_tokens: int,
        base_pf_block_table: List[int],
        base_pf_num_tokens: int,
        anchor_list: List[str],
        upstream_agent_id: str,
        temperature: float = 1.0,
        local_ref_mode: str = "no_check",
        consistency_threshold: float = 0.5,
        weight_threshold: float = 0.3,
    ) -> Tuple[List[int], int, List[int], int]:
        """Paged local-reference offset: use cross-agent delta pattern from
        historical anchors to improve the base-reference offset.

        approx = base_KV + Σ(w_k × delta_target_k) + Σ(w_k × (delta_target_k − delta_upstream_k))

        The cross-agent correction captures the stable KV difference pattern
        between agents across historical anchors and applies it to the current
        kv_reuse query.

        local_ref_mode controls the similarity gate:
          - "no_check":  always apply cross-delta correction
          - "cross_delta_consistency": apply only when cross-deltas across anchors
                                      have low relative std (stable pattern)
          - "weight_confidence": apply only when max softmax weight exceeds threshold

        Falls back to base-reference offset when no valid anchors exist.
        """
        ph_store = self.anchors.get(ph_id, {})

        # 1. Collect anchors with BOTH target and upstream agent deltas
        valid_anchors = []
        for msg in anchor_list:
            entry = ph_store.get(msg)
            if entry is None:
                continue
            if agent_id not in entry.agent_deltas or upstream_agent_id not in entry.agent_deltas:
                continue
            tgt_info = entry.agent_deltas[agent_id]
            up_info = entry.agent_deltas[upstream_agent_id]
            if entry.num_tokens < base_ph_num_tokens:
                continue
            if int(tgt_info.get("ph_delta_num_tokens", 0) or 0) < base_ph_num_tokens:
                continue
            if int(up_info.get("ph_delta_num_tokens", 0) or 0) < base_ph_num_tokens:
                continue
            valid_anchors.append((msg, entry))

        if not valid_anchors:
            logger.info(
                "[LOCAL_REF_FALLBACK] agent={} ph_id={} upstream={} reason=no_valid_anchors "
                "anchor_list_len={} ph_store_keys={}",
                agent_id, ph_id, upstream_agent_id, len(anchor_list), len(ph_store),
            )
            return self.offset_kv_cache(
                agent_id, ph_id, message,
                base_ph_block_table, base_ph_num_tokens,
                base_pf_block_table, base_pf_num_tokens,
                anchor_list, temperature,
            )

        # 2. Read base KV
        base_ph_key, base_ph_val = self.read_kv_from_blocks(
            base_ph_block_table, base_ph_num_tokens
        )

        # 3. Compute similarity weights
        sims_list = []
        for _, entry in valid_anchors:
            anchor_key = entry.ph_key_embedding[..., -base_ph_num_tokens:, :]
            if anchor_key.device != base_ph_key.device:
                anchor_key = anchor_key.to(base_ph_key.device, non_blocking=True)
            sims_list.append((base_ph_key - anchor_key).norm(2, dim=-2))
        sims = torch.stack(sims_list, dim=0)
        weights = torch.softmax(-sims.float() / temperature, dim=0).unsqueeze(-2)

        # 4. Compute base-reference offset (target agent's own deltas)
        ph_delta_sum_k = torch.zeros_like(base_ph_key)
        ph_delta_sum_v = torch.zeros_like(base_ph_val)
        # Also collect cross-deltas for the correction term
        cross_deltas_k = []
        cross_deltas_v = []
        for i, (msg, entry) in enumerate(valid_anchors):
            tgt_dk, tgt_dv = self.read_kv_from_blocks(
                entry.agent_deltas[agent_id]["ph_delta_blocks"], base_ph_num_tokens,
            )
            up_dk, up_dv = self.read_kv_from_blocks(
                entry.agent_deltas[upstream_agent_id]["ph_delta_blocks"], base_ph_num_tokens,
            )
            tgt_dk = tgt_dk[..., :base_ph_num_tokens, :]
            tgt_dv = tgt_dv[..., :base_ph_num_tokens, :]
            up_dk = up_dk[..., :base_ph_num_tokens, :]
            up_dv = up_dv[..., :base_ph_num_tokens, :]
            w = weights[i]
            ph_delta_sum_k += w * tgt_dk
            ph_delta_sum_v += w * tgt_dv
            cross_deltas_k.append(tgt_dk - up_dk)
            cross_deltas_v.append(tgt_dv - up_dv)

        # 5. Similarity gate: decide whether to apply cross-delta correction
        apply_cross_delta = True
        gate_info = ""

        if local_ref_mode == "cross_delta_consistency":
            # Check if cross-deltas are consistent across anchors (low variance → stable pattern)
            if len(cross_deltas_k) >= 2:
                stacked = torch.stack(cross_deltas_k, dim=0)  # [N, layers, heads, tokens, dim]
                mean_norm = stacked.float().mean(dim=0).norm(2, dim=-1).mean()
                std_norm = stacked.float().std(dim=0).norm(2, dim=-1).mean()
                relative_std = (std_norm / (mean_norm + 1e-8)).item()
                apply_cross_delta = relative_std < consistency_threshold
                gate_info = f"consistency relative_std={relative_std:.4f} threshold={consistency_threshold}"
            else:
                # Only 1 anchor: can't measure consistency, apply anyway
                gate_info = "consistency skip (single_anchor)"

        elif local_ref_mode == "weight_confidence":
            # Check if the best anchor weight is confident enough
            max_weight = weights.max().item()
            apply_cross_delta = max_weight > weight_threshold
            gate_info = f"weight_confidence max_weight={max_weight:.4f} threshold={weight_threshold}"

        else:
            gate_info = "no_check"

        if apply_cross_delta:
            # 6. Compute weighted cross-delta correction
            cross_sum_k = torch.zeros_like(base_ph_key)
            cross_sum_v = torch.zeros_like(base_ph_val)
            for i in range(len(valid_anchors)):
                w = weights[i]
                cross_sum_k += w * cross_deltas_k[i]
                cross_sum_v += w * cross_deltas_v[i]

            # 7. Apply: base + target_delta + cross_delta_correction
            new_ph_key = base_ph_key + ph_delta_sum_k.to(base_ph_key.dtype) + cross_sum_k.to(base_ph_key.dtype)
            new_ph_val = base_ph_val + ph_delta_sum_v.to(base_ph_val.dtype) + cross_sum_v.to(base_ph_val.dtype)
        else:
            # Gate rejected: use base-reference offset only (no cross-delta)
            new_ph_key = base_ph_key + ph_delta_sum_k.to(base_ph_key.dtype)
            new_ph_val = base_ph_val + ph_delta_sum_v.to(base_ph_val.dtype)

        new_ph_key[0] = base_ph_key[0]
        new_ph_val[0] = base_ph_val[0]

        new_ph_blocks = self.allocate_blocks_for_tokens(base_ph_num_tokens)
        self.write_kv_to_blocks(new_ph_blocks, new_ph_key, new_ph_val, base_ph_num_tokens)

        # 8. Prefix: same logic
        base_pf_key, base_pf_val = self.read_kv_from_blocks(
            base_pf_block_table, base_pf_num_tokens
        )

        has_pf = base_pf_num_tokens > 0 and all(
            upstream_agent_id in entry.agent_deltas
            and int(entry.agent_deltas[upstream_agent_id].get("pf_delta_num_tokens", 0) or 0) >= base_pf_num_tokens
            and int(entry.agent_deltas[agent_id].get("pf_delta_num_tokens", 0) or 0) >= base_pf_num_tokens
            for _, entry in valid_anchors
        )

        if has_pf and apply_cross_delta:
            pf_delta_k = torch.zeros_like(base_pf_key)
            pf_delta_v = torch.zeros_like(base_pf_val)
            pf_cross_k = torch.zeros_like(base_pf_key)
            pf_cross_v = torch.zeros_like(base_pf_val)
            for i, (msg, entry) in enumerate(valid_anchors):
                tgt_dk, tgt_dv = self.read_kv_from_blocks(
                    entry.agent_deltas[agent_id]["pf_delta_blocks"], base_pf_num_tokens,
                )
                up_dk, up_dv = self.read_kv_from_blocks(
                    entry.agent_deltas[upstream_agent_id]["pf_delta_blocks"], base_pf_num_tokens,
                )
                w = weights[i]
                pf_delta_k += w * tgt_dk
                pf_delta_v += w * tgt_dv
                pf_cross_k += w * (tgt_dk - up_dk)
                pf_cross_v += w * (tgt_dv - up_dv)
            new_pf_key = base_pf_key + pf_delta_k.to(base_pf_key.dtype) + pf_cross_k.to(base_pf_key.dtype)
            new_pf_val = base_pf_val + pf_delta_v.to(base_pf_val.dtype) + pf_cross_v.to(base_pf_val.dtype)
        else:
            # Prefix: base-reference only
            pf_delta_k = torch.zeros_like(base_pf_key)
            pf_delta_v = torch.zeros_like(base_pf_val)
            for i, (msg, entry) in enumerate(valid_anchors):
                dk, dv = self.read_kv_from_blocks(
                    entry.agent_deltas[agent_id]["pf_delta_blocks"], base_pf_num_tokens,
                )
                w = weights[i]
                pf_delta_k += w * dk
                pf_delta_v += w * dv
            new_pf_key = base_pf_key + pf_delta_k.to(base_pf_key.dtype)
            new_pf_val = base_pf_val + pf_delta_v.to(base_pf_val.dtype)

        new_pf_key[0] = base_pf_key[0]
        new_pf_val[0] = base_pf_val[0]

        new_pf_blocks = self.allocate_blocks_for_tokens(base_pf_num_tokens)
        self.write_kv_to_blocks(new_pf_blocks, new_pf_key, new_pf_val, base_pf_num_tokens)

        logger.info(
            "[LOCAL_REF:paged] {} | agent={} ph_id={} upstream={} "
            "valid_anchors={} apply_cross_delta={} gate={} ph_tokens={}",
            "APPLIED" if apply_cross_delta else "GATE_REJECTED",
            agent_id, ph_id, upstream_agent_id,
            len(valid_anchors), apply_cross_delta, gate_info, base_ph_num_tokens,
        )

        return new_ph_blocks, base_ph_num_tokens, new_pf_blocks, base_pf_num_tokens

    # ── Response embedding store ──────────────────────────────────────────────

    def store_response_embedding(
        self,
        ph_id: str,
        message: str,
        block_table: List[int],
        num_tokens: int,
        max_entries: int = 20,
    ) -> None:
        """Store actual response value KV for later response anchor prediction.

        This is separate from placeholder PagedAnchorEntry because:
        - entry.num_tokens = template placeholder size (e.g. 64 tokens)
        - actual response length varies (e.g. 150 tokens)
        Mixing them causes the length check in predict_as_anchor to always fail.
        """
        if num_tokens <= 0 or not block_table:
            return
        _, val = self.read_kv_from_blocks(block_table, num_tokens)
        emb_store = self.response_embeddings.setdefault(ph_id, {})
        cnt_store = self.response_token_counts.setdefault(ph_id, {})
        # Simple FIFO eviction when over capacity
        if len(emb_store) >= max_entries and message not in emb_store:
            oldest = next(iter(emb_store))
            del emb_store[oldest]
            del cnt_store[oldest]
        emb_store[message] = val.detach().cpu()
        cnt_store[message] = num_tokens

    def predict_as_anchor(
        self,
        ph_id: str,
        candidate_block_table: List[int],
        candidate_num_tokens: int,
        anchor_messages: List[str],
        top_p: float = 0.9,
        entropy_threshold: float = 0.5,
        max_compare_anchors: int = 64,
        use_response_embeddings: bool = False,
    ) -> Tuple[bool, List[int]]:
        """Decide whether to activate anchors based on KV similarity.

        Same logic as KVCOMMEngine.predict_as_anchor() but reads from blocks.

        Args:
            use_response_embeddings: When True, compare against stored response
                value embeddings (response_embeddings[ph_id]) instead of
                placeholder anchor entries (anchors[ph_id]).
                Must be True for response anchor prediction to avoid comparing
                template placeholder KV (64 tokens) against actual response KV
                (150+ tokens), which always fails the length check.

        Returns:
            (prob, activated_counts)

        Notes:
            - prob == True  => treat as new anchor (dense_prefill path)
            - prob == False => reuse existing anchors (kv_reuse path)
        """
        activated = [0] * len(anchor_messages)

        if use_response_embeddings:
            # Response anchor path: compare actual response KV vs stored response KVs.
            emb_store = self.response_embeddings.get(ph_id, {})
            cnt_store = self.response_token_counts.get(ph_id, {})

            available_indices = []
            available_vals = []
            for i, msg in enumerate(anchor_messages):
                stored_val = emb_store.get(msg)
                stored_cnt = cnt_store.get(msg, 0)
                if stored_val is not None and stored_cnt >= candidate_num_tokens:
                    available_indices.append(i)
                    available_vals.append(stored_val)

            if max_compare_anchors > 0 and len(available_vals) > max_compare_anchors:
                available_indices = available_indices[-max_compare_anchors:]
                available_vals = available_vals[-max_compare_anchors:]

            if len(anchor_messages) == 0:
                reason = "no_anchor_history"
            elif len(available_vals) == 0:
                reason = "no_length_eligible_anchor"
            elif len(available_vals) == 1:
                reason = "single_length_eligible_anchor"
            else:
                reason = "insufficient_anchors"

            if len(available_vals) <= 1:
                logger.info(
                    "[ANCHOR_PREDICT:paged] ph_id={} anchor_messages={} available_entries={} "
                    "candidate_num_tokens={} decision=dense_prefill reason={} mode=response",
                    ph_id, len(anchor_messages), len(available_vals),
                    candidate_num_tokens, reason,
                )
                return True, activated

            _, cand_val = self.read_kv_from_blocks(candidate_block_table, candidate_num_tokens)

            diff_list = []
            for stored_val in available_vals:
                anchor_val = stored_val[..., :candidate_num_tokens, :]
                if anchor_val.device != cand_val.device:
                    anchor_val = anchor_val.to(cand_val.device, non_blocking=True)
                diff_list.append((cand_val - anchor_val).norm(2, dim=(0, 1, 2, 3)))

        else:
            # Placeholder anchor path (original logic).
            ph_store = self.anchors.get(ph_id, {})

            available_indices = []
            available_entries = []
            for i, msg in enumerate(anchor_messages):
                entry = ph_store.get(msg)
                if entry is not None and entry.num_tokens >= candidate_num_tokens:
                    available_indices.append(i)
                    available_entries.append(entry)

            if max_compare_anchors > 0 and len(available_entries) > max_compare_anchors:
                available_indices = available_indices[-max_compare_anchors:]
                available_entries = available_entries[-max_compare_anchors:]

            if len(anchor_messages) == 0:
                reason = "no_anchor_history"
            elif len(available_entries) == 0:
                reason = "no_length_eligible_anchor"
            elif len(available_entries) == 1:
                reason = "single_length_eligible_anchor"
            else:
                reason = "insufficient_anchors"

            if len(available_entries) <= 1:
                logger.info(
                    "[ANCHOR_PREDICT:paged] ph_id={} anchor_messages={} available_entries={} "
                    "candidate_num_tokens={} decision=dense_prefill reason={}",
                    ph_id, len(anchor_messages), len(available_entries),
                    candidate_num_tokens, reason,
                )
                return True, activated

            _, cand_val = self.read_kv_from_blocks(candidate_block_table, candidate_num_tokens)

            diff_list = []
            for entry in available_entries:
                anchor_val = entry.ph_value_embedding[..., :candidate_num_tokens, :]
                if anchor_val.device != cand_val.device:
                    anchor_val = anchor_val.to(cand_val.device, non_blocking=True)
                diff_list.append((cand_val - anchor_val).norm(2, dim=(0, 1, 2, 3)))

            available_vals = available_entries  # alias for unified code below

        diff = torch.stack(diff_list, dim=0)
        sim = torch.softmax(-diff.float(), dim=0)

        # Entropy check
        entropy = -(sim * (sim + 1e-40).log2()).sum()
        threshold = entropy_threshold * torch.log2(torch.tensor(float(sim.shape[0])))
        if entropy > threshold:
            logger.info(
                "[ANCHOR_PREDICT:paged] ph_id={} anchor_messages={} available_entries={} "
                "candidate_num_tokens={} entropy={} threshold={} decision=dense_prefill reason=high_entropy",
                ph_id, len(anchor_messages), len(available_vals),
                candidate_num_tokens, float(entropy.item()), float(threshold.item()),
            )
            return True, activated

        # Top-p selection
        sorted_sim, sorted_idx = torch.sort(sim, descending=True)
        cum = torch.cumsum(sorted_sim, dim=0)
        cutoff_candidates = (cum < top_p).nonzero(as_tuple=True)[0]
        cutoff = cutoff_candidates[-1] if len(cutoff_candidates) > 0 else len(sorted_sim) - 1
        selected = sorted_idx[:cutoff + 1]

        for s in selected:
            idx = available_indices[s.item()]
            if idx < len(activated):
                activated[idx] += 1

        logger.info(
            "[ANCHOR_PREDICT:paged] ph_id={} anchor_messages={} available_entries={} "
            "candidate_num_tokens={} selected={} decision=kv_reuse",
            ph_id, len(anchor_messages), len(available_vals),
            candidate_num_tokens, int(len(selected)),
        )
        return False, activated

    def get_seq_block_table(self, seq: Sequence) -> List[int]:
        """Get block table from a nano-vllm Sequence object."""
        return list(seq.block_table)

    def get_seq_num_tokens(self, seq: Sequence) -> int:
        """Get number of tokens from a nano-vllm Sequence object."""
        return len(seq)

    def fork_block_table(self, block_table: List[int]) -> List[int]:
        """Create a shared copy of a block table (increment ref counts)."""
        self.increment_ref(block_table)
        return list(block_table)

    def free_anchor(self, ph_id: str, message: str) -> None:
        """Free an anchor's blocks."""
        ph_store = self.anchors.get(ph_id, {})
        entry = ph_store.pop(message, None)
        if entry is None:
            return

        # Free base blocks
        self.free_blocks(entry.block_table)

        # Free delta blocks for all agents
        for agent_id, delta_info in entry.agent_deltas.items():
            self.free_blocks(delta_info.get("ph_delta_blocks", []))
            self.free_blocks(delta_info.get("pf_delta_blocks", []))

        # Clean up activation tracking
        info_store = self.anchor_info.get(ph_id, {})
        info_store.pop(message, None)

    def evict_anchor(self, ph_id: str, window_length: int = 5) -> bool:
        """Evict the least-activated anchor within a window for a given ph_id.

        Mirrors KVCOMMEngine.update_anchor(): looks at the oldest `effective_window`
        anchors (by insertion order) and removes the one with the lowest
        activation count, freeing its blocks back to the pool.

        The effective window is sampled uniformly from [5, 8] each call to
        introduce randomness in eviction candidate selection, which avoids
        repeatedly evicting from the same fixed prefix of the anchor pool.

        Returns True if an anchor was evicted, False otherwise.
        """
        ph_store = self.anchors.get(ph_id, {})
        if not ph_store:
            return False

        info_store = self.anchor_info.get(ph_id, {})
        # Randomly choose effective window size in [5, 8] for varied eviction
        effective_window = random.randint(5, 8)
        messages = list(ph_store.keys())[:effective_window]
        if not messages:
            return False

        # Find the message with the lowest activation count
        min_count = float("inf")
        min_msg = messages[0]
        for msg in messages:
            count = info_store.get(msg, 0)
            if count < min_count:
                min_count = count
                min_msg = msg

        logger.info(
            "[ANCHOR_EVICT:paged] ph_id={} evicted_message={} activation_count={} "
            "effective_window={} remaining_anchors={}",
            ph_id, min_msg[:80], min_count, effective_window, len(ph_store) - 1,
        )
        self.free_anchor(ph_id, min_msg)
        return True

    def evict_until_free(self, num_blocks_needed: int, window_length: int = 5) -> bool:
        """Evict anchors across all ph_ids until enough free blocks are available.

        Returns True if enough blocks were freed, False otherwise.
        """
        while len(self.block_manager.free_block_ids) < num_blocks_needed:
            # Find the ph_id with the most anchors to evict from
            evicted = False
            best_ph_id = None
            best_count = 0
            for ph_id, ph_store in self.anchors.items():
                if len(ph_store) > best_count:
                    best_count = len(ph_store)
                    best_ph_id = ph_id
            if best_ph_id is not None and best_count > 0:
                evicted = self.evict_anchor(best_ph_id, window_length)
            if not evicted:
                return False
        return True

    def evict_proactive(
        self,
        ph_id: Optional[str] = None,
        free_ratio_threshold: float = 0.15,
        window_length: int = 5,
    ) -> int:
        """Proactively evict anchors when the free pool fraction falls below threshold.

        Mirrors gpt_chat's update_anchor: tries the given ph_id first (same as
        the anchor being written), then falls back to the ph_id with the most
        anchors globally — so the pool is kept healthy before delta blocks are
        allocated, not only after it is completely exhausted.

        Returns the number of anchors evicted.
        """
        total = len(self.block_manager.blocks)
        if total == 0:
            return 0
        evicted = 0
        while len(self.block_manager.free_block_ids) / total < free_ratio_threshold:
            # 1. Same ph_id first (aligned with gpt_chat update_anchor)
            if ph_id and self.anchors.get(ph_id):
                if self.evict_anchor(ph_id, window_length):
                    evicted += 1
                    continue
            # 2. Global fallback: ph_id with the most anchors
            best_ph_id = None
            best_count = 0
            for pid, ph_store in self.anchors.items():
                if len(ph_store) > best_count:
                    best_count = len(ph_store)
                    best_ph_id = pid
            if best_ph_id is not None and self.evict_anchor(best_ph_id, window_length):
                evicted += 1
            else:
                break  # nothing left to evict
        if evicted > 0:
            logger.info(
                "[PROACTIVE_EVICT] ph_id={} evicted={} free_blocks={} total_blocks={} free_ratio={:.3f}",
                ph_id,
                evicted,
                len(self.block_manager.free_block_ids),
                total,
                len(self.block_manager.free_block_ids) / total,
            )
        return evicted

    def get_memory_stats(self) -> Dict[str, int]:
        """Return memory usage statistics."""
        total = len(self.block_manager.blocks)
        free = len(self.block_manager.free_block_ids)
        used = len(self.block_manager.used_block_ids)
        return {
            "total_blocks": total,
            "free_blocks": free,
            "used_blocks": used,
            "block_size": self.block_size,
            "num_anchors": sum(len(v) for v in self.anchors.values()),
        }
