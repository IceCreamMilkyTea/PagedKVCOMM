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
        self._lock = threading.Lock()

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
        """Allocate fresh blocks from the free pool for a given number of tokens."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        block_ids = []
        for _ in range(num_blocks_needed):
            if not self.block_manager.free_block_ids:
                raise RuntimeError(
                    f"Not enough free blocks: need {num_blocks_needed}, "
                    f"have {len(self.block_manager.free_block_ids)}"
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

            self.increment_ref(block_table)
            ph_store[message] = PagedAnchorEntry(
                block_table=list(block_table),
                num_tokens=num_tokens,
                ph_key_embedding=ph_key,
                ph_value_embedding=ph_val,
            )
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
    ) -> None:
        """Store an anchor entry using block references + delta.

        Steps:
          1. Read base KV from blocks → store embedding for similarity computation
          2. Read real KV from blocks → compute delta = real - base
          3. Allocate new blocks for delta → write delta to blocks
          4. Store block references (not tensors) in anchor entry

        The base's block_table is shared (via ref counting) so multiple agents
        referencing the same prefix share physical blocks → zero-copy.
        """
        # Read base KV for similarity (placeholder portion)
        base_ph_key, base_ph_val = self.read_kv_from_blocks(base_block_table, base_num_tokens)

        with self._lock:
            ph_store = self.anchors.setdefault(ph_id, {})

            if message not in ph_store:
                # New anchor: store base embedding + block reference
                self.increment_ref(base_block_table)
                entry = PagedAnchorEntry(
                    block_table=list(base_block_table),
                    num_tokens=base_num_tokens,
                    ph_key_embedding=base_ph_key,
                    ph_value_embedding=base_ph_val,
                )
                ph_store[message] = entry
            else:
                entry = ph_store[message]

        # Compute placeholder delta: real - base
        real_ph_key, real_ph_val = self.read_kv_from_blocks(real_block_table, real_num_tokens)
        if real_ph_key.shape[2] != real_num_tokens or real_ph_key.shape[2] == 0:
            raise ValueError(
                f"set_anchor: real_ph_key has {real_ph_key.shape[2]} tokens at dim 2, "
                f"expected {real_num_tokens}. real_block_table may be empty or misaligned."
            )
        ph_key_delta = real_ph_key - base_ph_key[..., :real_num_tokens, :]
        ph_val_delta = real_ph_val - base_ph_val[..., :real_num_tokens, :]

        # Allocate blocks for placeholder delta and write
        ph_delta_blocks = self.allocate_blocks_for_tokens(real_num_tokens)
        self.write_kv_to_blocks(ph_delta_blocks, ph_key_delta, ph_val_delta, real_num_tokens)

        # Compute prefix delta: real - base
        if real_prefix_num_tokens <= 0 or base_prefix_num_tokens <= 0:
            return
        real_pf_key, real_pf_val = self.read_kv_from_blocks(
            real_prefix_block_table, real_prefix_num_tokens
        )
        base_pf_key, base_pf_val = self.read_kv_from_blocks(
            base_prefix_block_table, base_prefix_num_tokens
        )
        pf_tokens = min(real_pf_key.shape[2], base_pf_key.shape[2], real_prefix_num_tokens, base_prefix_num_tokens)
        if pf_tokens <= 0:
            return
        pf_key_delta = real_pf_key[..., :pf_tokens, :] - base_pf_key[..., :pf_tokens, :]
        pf_val_delta = real_pf_val[..., :pf_tokens, :] - base_pf_val[..., :pf_tokens, :]

        pf_delta_blocks = self.allocate_blocks_for_tokens(pf_tokens)
        self.write_kv_to_blocks(pf_delta_blocks, pf_key_delta, pf_val_delta, pf_tokens)

        # Store delta block references for this agent
        with self._lock:
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
        temperature: float = 1.0,
    ) -> Tuple[List[int], int, List[int], int, Dict[str, Any]]:
        """Apply weighted cross-agent deltas using a local (upstream) reference.

        Instead of reconstructing from global base:
            KV(x|P_j) ≈ KV(x|base) + Σ wᵢ·Δ(base→Pᵢ)

        This uses the closest upstream agent's KV as reference:
            KV(x|P_j) ≈ KV(x|P_up) + Σ wᵢ·(Δ(base→Pᵢ) − Δ(base→P_up))

        The upstream agent is selected as the agent (other than ``agent_id``)
        whose delta is most similar to the current agent across the valid
        anchors.  If no upstream agent delta is available, this falls back to
        the standard ``offset_kv_cache`` (global base reference).

        Returns:
            (new_ph_block_table, ph_num_tokens, new_pf_block_table, pf_num_tokens, local_ref_info)
            local_ref_info keys:
              - local_ref_used (bool): True if local reference was applied
              - upstream_agent_id (str|None): selected upstream agent
              - upstream_dist (float): L2 distance to upstream
              - fallback_reason (str|None): reason if fell back to global base
              - num_candidate_upstreams (int): number of candidate upstream agents
        """
        ph_store = self.anchors.get(ph_id, {})
        if not ph_store or not anchor_list:
            self.increment_ref(base_ph_block_table)
            self.increment_ref(base_pf_block_table)
            return (base_ph_block_table, base_ph_num_tokens,
                    base_pf_block_table, base_pf_num_tokens,
                    {"local_ref_used": False, "upstream_agent_id": None,
                     "upstream_dist": 0.0, "fallback_reason": "no_anchors_or_anchor_list",
                     "num_candidate_upstreams": 0})

        # Collect valid anchors (same criteria as offset_kv_cache).
        valid_anchors: List[Tuple[str, PagedAnchorEntry]] = []
        for msg in anchor_list:
            entry = ph_store.get(msg)
            if entry is None:
                continue
            delta_info = entry.agent_deltas.get(agent_id)
            if delta_info is None:
                continue
            if entry.num_tokens < base_ph_num_tokens:
                continue
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
            return (base_ph_block_table, base_ph_num_tokens,
                    base_pf_block_table, base_pf_num_tokens,
                    {"local_ref_used": False, "upstream_agent_id": None,
                     "upstream_dist": 0.0, "fallback_reason": "no_valid_anchors",
                     "num_candidate_upstreams": 0})

        # --- Find the best upstream agent ---
        # Collect all agent IDs (excluding current) that have deltas in *every*
        # valid anchor so we can compute cross-agent offsets consistently.
        candidate_upstream_ids: Optional[set] = None
        for _, entry in valid_anchors:
            entry_agents = set()
            for aid, d in entry.agent_deltas.items():
                if aid == agent_id:
                    continue
                # Upstream must also cover the required token spans.
                up_ph = int(d.get("ph_delta_num_tokens", 0) or 0)
                if up_ph < base_ph_num_tokens:
                    continue
                if base_pf_num_tokens > 0:
                    up_pf = int(d.get("pf_delta_num_tokens", 0) or 0)
                    if up_pf < base_pf_num_tokens:
                        continue
                entry_agents.add(aid)
            if candidate_upstream_ids is None:
                candidate_upstream_ids = entry_agents
            else:
                candidate_upstream_ids &= entry_agents

        if not candidate_upstream_ids:
            # No common upstream agent available across all anchors → fallback.
            logger.info(
                "[LOCAL_REF] ph_id={} agent={} no common upstream agent, "
                "falling back to global base offset",
                ph_id, agent_id,
            )
            ph, phn, pf, pfn = self.offset_kv_cache(
                agent_id=agent_id,
                ph_id=ph_id,
                message=message,
                base_ph_block_table=base_ph_block_table,
                base_ph_num_tokens=base_ph_num_tokens,
                base_pf_block_table=base_pf_block_table,
                base_pf_num_tokens=base_pf_num_tokens,
                anchor_list=anchor_list,
                temperature=temperature,
            )
            return (ph, phn, pf, pfn,
                    {"local_ref_used": False, "upstream_agent_id": None,
                     "upstream_dist": 0.0, "fallback_reason": "no_common_upstream",
                     "num_candidate_upstreams": 0})

        # Pick the upstream agent whose average delta is closest to the
        # current agent's delta (smallest L2 distance across anchors).
        base_ph_key, base_ph_val = self.read_kv_from_blocks(base_ph_block_table, base_ph_num_tokens)
        base_pf_key, base_pf_val = self.read_kv_from_blocks(base_pf_block_table, base_pf_num_tokens)

        best_upstream_id: Optional[str] = None
        best_dist = float("inf")
        for up_id in candidate_upstream_ids:
            dist_acc = 0.0
            for _, entry in valid_anchors:
                cur_delta = entry.agent_deltas[agent_id]
                up_delta = entry.agent_deltas[up_id]
                cur_dk, _ = self.read_kv_from_blocks(
                    cur_delta["ph_delta_blocks"], base_ph_num_tokens,
                )
                up_dk, _ = self.read_kv_from_blocks(
                    up_delta["ph_delta_blocks"], base_ph_num_tokens,
                )
                dist_acc += (cur_dk[..., :base_ph_num_tokens, :] - up_dk[..., :base_ph_num_tokens, :]).norm().item()
            if dist_acc < best_dist:
                best_dist = dist_acc
                best_upstream_id = up_id

        if best_upstream_id is None:
            # Should not happen given the check above, but guard anyway.
            ph, phn, pf, pfn = self.offset_kv_cache(
                agent_id=agent_id, ph_id=ph_id, message=message,
                base_ph_block_table=base_ph_block_table, base_ph_num_tokens=base_ph_num_tokens,
                base_pf_block_table=base_pf_block_table, base_pf_num_tokens=base_pf_num_tokens,
                anchor_list=anchor_list, temperature=temperature,
            )
            return (ph, phn, pf, pfn,
                    {"local_ref_used": False, "upstream_agent_id": None,
                     "upstream_dist": 0.0, "fallback_reason": "no_best_upstream",
                     "num_candidate_upstreams": len(candidate_upstream_ids)})

        logger.info(
            "[LOCAL_REF] ph_id={} agent={} selected upstream={} dist={:.4f} "
            "num_valid_anchors={}",
            ph_id, agent_id, best_upstream_id, best_dist, len(valid_anchors),
        )

        # --- Compute similarity weights (same as global path) ---
        sims_list = []
        for _, entry in valid_anchors:
            anchor_key = entry.ph_key_embedding[..., -base_ph_num_tokens:, :]
            if anchor_key.device != base_ph_key.device:
                anchor_key = anchor_key.to(base_ph_key.device, non_blocking=True)
            sims_list.append((base_ph_key - anchor_key).norm(2, dim=-2))
        sims = torch.stack(sims_list, dim=0)
        weights = torch.softmax(-sims.float() / temperature, dim=0).unsqueeze(-2)

        # --- Reconstruct local reference KV (base + upstream delta) ---
        # Read upstream agent's delta from the first valid anchor to build the
        # reference.  We use a weighted average of upstream deltas across all
        # anchors (same weights) so the reference is consistent with the
        # blending.
        up_ph_delta_sum_k = torch.zeros_like(base_ph_key)
        up_ph_delta_sum_v = torch.zeros_like(base_ph_val)
        up_pf_delta_sum_k = torch.zeros_like(base_pf_key)
        up_pf_delta_sum_v = torch.zeros_like(base_pf_val)

        for i, (_, entry) in enumerate(valid_anchors):
            up_delta = entry.agent_deltas[best_upstream_id]
            dk, dv = self.read_kv_from_blocks(up_delta["ph_delta_blocks"], base_ph_num_tokens)
            w = weights[i]
            up_ph_delta_sum_k += w * dk[..., :base_ph_num_tokens, :]
            up_ph_delta_sum_v += w * dv[..., :base_ph_num_tokens, :]
            if base_pf_num_tokens > 0:
                dk_pf, dv_pf = self.read_kv_from_blocks(up_delta["pf_delta_blocks"], base_pf_num_tokens)
                up_pf_delta_sum_k += w * dk_pf
                up_pf_delta_sum_v += w * dv_pf

        # Local reference = base + upstream_delta
        local_ref_ph_key = base_ph_key + up_ph_delta_sum_k.to(base_ph_key.dtype)
        local_ref_ph_val = base_ph_val + up_ph_delta_sum_v.to(base_ph_val.dtype)
        local_ref_pf_key = base_pf_key + up_pf_delta_sum_k.to(base_pf_key.dtype)
        local_ref_pf_val = base_pf_val + up_pf_delta_sum_v.to(base_pf_val.dtype)

        # --- Weighted cross-delta: Δ_cross = Δ_current − Δ_upstream ---
        cross_ph_delta_k = torch.zeros_like(base_ph_key)
        cross_ph_delta_v = torch.zeros_like(base_ph_val)
        cross_pf_delta_k = torch.zeros_like(base_pf_key)
        cross_pf_delta_v = torch.zeros_like(base_pf_val)

        for i, (_, entry) in enumerate(valid_anchors):
            cur_delta = entry.agent_deltas[agent_id]
            up_delta = entry.agent_deltas[best_upstream_id]

            cur_dk, cur_dv = self.read_kv_from_blocks(cur_delta["ph_delta_blocks"], base_ph_num_tokens)
            up_dk, up_dv = self.read_kv_from_blocks(up_delta["ph_delta_blocks"], base_ph_num_tokens)
            w = weights[i]
            cross_ph_delta_k += w * (cur_dk[..., :base_ph_num_tokens, :] - up_dk[..., :base_ph_num_tokens, :])
            cross_ph_delta_v += w * (cur_dv[..., :base_ph_num_tokens, :] - up_dv[..., :base_ph_num_tokens, :])

            if base_pf_num_tokens > 0:
                cur_dk_pf, cur_dv_pf = self.read_kv_from_blocks(cur_delta["pf_delta_blocks"], base_pf_num_tokens)
                up_dk_pf, up_dv_pf = self.read_kv_from_blocks(up_delta["pf_delta_blocks"], base_pf_num_tokens)
                cross_pf_delta_k += w * (cur_dk_pf - up_dk_pf)
                cross_pf_delta_v += w * (cur_dv_pf - up_dv_pf)

        # --- Apply cross-delta to local reference ---
        new_ph_key = local_ref_ph_key + cross_ph_delta_k.to(local_ref_ph_key.dtype)
        new_ph_val = local_ref_ph_val + cross_ph_delta_v.to(local_ref_ph_val.dtype)
        new_ph_key[0] = base_ph_key[0]  # Keep layer 0 unchanged (KVCOMM convention)
        new_ph_val[0] = base_ph_val[0]

        new_ph_blocks = self.allocate_blocks_for_tokens(base_ph_num_tokens)
        self.write_kv_to_blocks(new_ph_blocks, new_ph_key, new_ph_val, base_ph_num_tokens)

        new_pf_key = local_ref_pf_key + cross_pf_delta_k.to(local_ref_pf_key.dtype)
        new_pf_val = local_ref_pf_val + cross_pf_delta_v.to(local_ref_pf_val.dtype)
        new_pf_key[0] = base_pf_key[0]
        new_pf_val[0] = base_pf_val[0]

        new_pf_blocks = self.allocate_blocks_for_tokens(base_pf_num_tokens)
        self.write_kv_to_blocks(new_pf_blocks, new_pf_key, new_pf_val, base_pf_num_tokens)

        return (new_ph_blocks, base_ph_num_tokens, new_pf_blocks, base_pf_num_tokens,
                {"local_ref_used": True, "upstream_agent_id": best_upstream_id,
                 "upstream_dist": best_dist,
                 "fallback_reason": None,
                 "num_candidate_upstreams": len(candidate_upstream_ids)})

    def predict_as_anchor(
        self,
        ph_id: str,
        candidate_block_table: List[int],
        candidate_num_tokens: int,
        anchor_messages: List[str],
        top_p: float = 0.9,
        entropy_threshold: float = 0.5,
        max_compare_anchors: int = 64,
    ) -> Tuple[bool, List[int]]:
        """Decide whether to activate anchors based on KV similarity.

        Same logic as KVCOMMEngine.predict_as_anchor() but reads from blocks.

        Returns:
            (prob, activated_counts)

        Notes:
            - prob == True  => treat as new anchor (dense_prefill path)
            - prob == False => reuse existing anchors (kv_reuse path)
        """
        ph_store = self.anchors.get(ph_id, {})
        activated = [0] * len(anchor_messages)

        available_indices = []
        available_entries = []
        for i, msg in enumerate(anchor_messages):
            entry = ph_store.get(msg)
            if entry is not None and entry.num_tokens >= candidate_num_tokens:
                available_indices.append(i)
                available_entries.append(entry)

        # Compare only a bounded recent subset to keep predict path memory stable.
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
                ph_id,
                len(anchor_messages),
                len(available_entries),
                candidate_num_tokens,
                reason,
            )
            return True, activated

        # Read candidate KV
        _, cand_val = self.read_kv_from_blocks(candidate_block_table, candidate_num_tokens)

        # Stream anchor distance computation to avoid large temporary tensors.
        diff_list = []
        for entry in available_entries:
            anchor_val = entry.ph_value_embedding[..., :candidate_num_tokens, :]
            if anchor_val.device != cand_val.device:
                anchor_val = anchor_val.to(cand_val.device, non_blocking=True)
            diff_list.append((cand_val - anchor_val).norm(2, dim=(0, 1, 2, 3)))
        diff = torch.stack(diff_list, dim=0)
        sim = torch.softmax(-diff.float(), dim=0)

        # Entropy check
        entropy = -(sim * (sim + 1e-40).log2()).sum()
        threshold = entropy_threshold * torch.log2(torch.tensor(float(sim.shape[0])))
        if entropy > threshold:
            logger.info(
                "[ANCHOR_PREDICT:paged] ph_id={} anchor_messages={} available_entries={} "
                "candidate_num_tokens={} entropy={} threshold={} decision=dense_prefill reason=high_entropy",
                ph_id,
                len(anchor_messages),
                len(available_entries),
                candidate_num_tokens,
                float(entropy.item()),
                float(threshold.item()),
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
            ph_id,
            len(anchor_messages),
            len(available_entries),
            candidate_num_tokens,
            int(len(selected)),
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
