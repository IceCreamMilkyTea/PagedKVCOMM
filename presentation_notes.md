# PagedKVCOMM Presentation Notes

---

## Part 1: Architecture Diagrams

### 1.1 nano-vllm Architecture (Inference Engine)

```
┌──────────────────────────────────────────────────────────────────┐
│                         LLM (user API)                           │
│                    llm.generate(prompts)                          │
└──────────────────────────┬───────────────────────────────────────┘
                           │ inherits
┌──────────────────────────▼───────────────────────────────────────┐
│                       LLMEngine                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │  Tokenizer   │  │  Scheduler   │  │     ModelRunner          │ │
│  │ (encode/     │  │ (WAITING →   │  │  (GPU forward pass)      │ │
│  │  decode)     │  │  RUNNING →   │  │                          │ │
│  │              │  │  FINISHED)   │  │  ┌────────────────────┐  │ │
│  └─────────────┘  │              │  │  │ Model (Llama/Qwen) │  │ │
│                    │  ┌────────┐  │  │  │  ┌──────────────┐  │  │ │
│                    │  │ Block  │  │  │  │  │  Attention    │  │  │ │
│                    │  │Manager │  │  │  │  │ (flash_attn + │  │  │ │
│                    │  │(alloc/ │  │  │  │  │  paged KV)   │  │  │ │
│                    │  │ dealloc│  │  │  │  └──────────────┘  │  │ │
│                    │  │ prefix │  │  │  │  ┌──────────────┐  │  │ │
│                    │  │ cache) │  │  │  │  │  Sampler     │  │  │ │
│                    │  └────────┘  │  │  │  │ (greedy/temp)│  │  │ │
│                    └──────────────┘  │  │  └──────────────┘  │  │ │
│                                      │  └────────────────────┘  │ │
│                                      │                          │ │
│                                      │  ┌────────────────────┐  │ │
│                                      │  │  KV Cache Pool     │  │ │
│                                      │  │ [2, L, num_blocks, │  │ │
│                                      │  │  block_size, H, D] │  │ │
│                                      │  └────────────────────┘  │ │
│                                      └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

Inference Loop (per step):
  1. Scheduler.schedule() → select seqs, allocate blocks
  2. ModelRunner.run()    → prepare inputs → forward pass → sample
  3. Scheduler.postprocess() → append tokens, check EOS, free blocks
```

### 1.2 KVCOMM Architecture (Multi-Agent KV Reuse Framework)

```
┌────────────────────────────────────────────────────────────────────┐
│                          Graph (DAG Orchestrator)                   │
│                                                                     │
│   ┌──────────┐   spatial    ┌──────────┐   spatial    ┌──────────┐ │
│   │  Node A  │─────────────>│  Node B  │─────────────>│ Decision │ │
│   │(Analyze) │              │(MathSolv)│              │ (Final)  │ │
│   └────┬─────┘              └────┬─────┘              └──────────┘ │
│        │ temporal                 │ temporal                        │
│        ▼                         ▼                                  │
│   ┌──────────┐              ┌──────────┐                           │
│   │  Node A' │              │  Node B' │    (next round)           │
│   └──────────┘              └──────────┘                           │
└───────────────────────────┬────────────────────────────────────────┘
                            │ each node calls
┌───────────────────────────▼────────────────────────────────────────┐
│                      LLMRegistry                                    │
│         ┌─────────────────┬─────────────────┐                       │
│         │                 │                  │                       │
│    ┌────▼────┐     ┌──────▼──────┐    ┌─────▼──────┐               │
│    │ GPTChat │     │   LLMChat   │    │PagedLLMChat│               │
│    │(OpenAI) │     │(HF+Dynamic  │    │(nano-vllm +│               │
│    │         │     │   Cache)    │    │PagedKVCOMM)│               │
│    └─────────┘     └──────┬──────┘    └─────┬──────┘               │
│                           │                  │                       │
│                    ┌──────▼──────┐    ┌──────▼───────┐              │
│                    │ KVCOMMEngine│    │PagedKVCOMM   │              │
│                    │(DynamicCache│    │   Engine      │              │
│                    │  anchors)   │    │(block-based   │              │
│                    └─────────────┘    │  anchors)     │              │
│                                      └───────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 PagedKVCOMM Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       PagedLLMChat                                   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    Shared Resources (class-level)               │  │
│  │  _shared_engine   = LLMEngine (nano-vllm)                      │  │
│  │  _shared_tokenizer                                              │  │
│  │  _paged_kv_engine = PagedKVCOMMEngine                           │  │
│  │  _shared_kv_cache_memory (per-agent prefix stores)              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─────────────────── Generation Flow ───────────────────────────┐  │
│  │                                                                │  │
│  │  1. prepare_prefix_kv_segments()                               │  │
│  │     → Prefill base template → store prefix blocks              │  │
│  │     → Register base anchors for each placeholder               │  │
│  │                                                                │  │
│  │  2. generate_with_kv_reuse() / generate_for_agent()            │  │
│  │     ┌──────────────────────────────────────────────┐           │  │
│  │     │ _prepare_kv_reuse_prefix_blocks()            │           │  │
│  │     │  For each placeholder span:                  │           │  │
│  │     │   Path 1: CRS (Current-Round Sharing)        │           │  │
│  │     │     → base + (delta_upstream - cross_delta)  │           │  │
│  │     │   Path 2: KVCOMM anchor matching             │           │  │
│  │     │     → offset_kv_cache(weighted delta blend)  │           │  │
│  │     │   Path 3: Fallback → use base blocks         │           │  │
│  │     └──────────────────────────────────────────────┘           │  │
│  │                       │                                        │  │
│  │                       ▼                                        │  │
│  │     ┌──────────────────────────────────────────────┐           │  │
│  │     │ _generate_tokens()                           │           │  │
│  │     │  → Sequence(token_ids, cached_prefix_blocks) │           │  │
│  │     │  → Scheduler.add(seq)                        │           │  │
│  │     │  → Loop: schedule → model_runner.run → post  │           │  │
│  │     │  → Pin blocks before deallocation             │           │  │
│  │     │  → Return (tokens, ttft, seq)                │           │  │
│  │     └──────────────────────────────────────────────┘           │  │
│  │                       │                                        │  │
│  │                       ▼                                        │  │
│  │     ┌──────────────────────────────────────────────┐           │  │
│  │     │ set_anchor() via PagedKVCOMMEngine            │           │  │
│  │     │  → Read real KV from generation blocks        │           │  │
│  │     │  → Compute delta = real_KV - base_KV          │           │  │
│  │     │  → Allocate new blocks, write delta           │           │  │
│  │     │  → Store PagedAnchorEntry atomically          │           │  │
│  │     └──────────────────────────────────────────────┘           │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─────────────────── PagedKVCOMMEngine ─────────────────────────┐  │
│  │                                                                │  │
│  │  kv_cache ←──── SAME tensor as ModelRunner's KV cache          │  │
│  │  block_manager ← SAME instance as Scheduler's BlockManager     │  │
│  │                                                                │  │
│  │  anchors: { ph_id → { message → PagedAnchorEntry } }          │  │
│  │                                                                │  │
│  │  PagedAnchorEntry:                                             │  │
│  │    block_table: [b1, b2, b3]     ← refs to kv_cache blocks    │  │
│  │    num_tokens: 48                                               │  │
│  │    ph_key_embedding: tensor      ← for similarity matching     │  │
│  │    ph_value_embedding: tensor                                   │  │
│  │    agent_deltas: {                                              │  │
│  │      agent_0: { ph_delta_blocks, pf_delta_blocks, ... }        │  │
│  │      agent_1: { ... }                                           │  │
│  │    }                                                            │  │
│  │                                                                │  │
│  │  Key Operations:                                                │  │
│  │    read_kv_from_blocks()  → reconstruct tensors from blocks     │  │
│  │    write_kv_to_blocks()   → scatter via triton kernel           │  │
│  │    set_anchor()           → store delta as block refs           │  │
│  │    offset_kv_cache()      → weighted delta blending             │  │
│  │    allocate/free_blocks() → pool management with eviction       │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: PPT Bullet Points

### Slide 1: Overview — What is KVCOMM?

- Multi-agent LLM framework where agents share a **common prompt template** with varying placeholders
- Core insight: agents' KV caches for shared template portions are **similar but not identical**
- KVCOMM stores historical KV as "anchors" and approximates new KV via **weighted delta blending**
- Avoids redundant prefill computation → reduces **Time-To-First-Token (TTFT)**

### Slide 2: nano-vllm — Lightweight Inference Engine

- Minimalist vLLM-style engine: **LLMEngine → Scheduler → ModelRunner → Model**
- **Paged Attention**: KV cache divided into fixed-size blocks (default 256 tokens)
  - Zero fragmentation — blocks allocated/freed dynamically
  - Prefix caching via semantic hash matching (xxHash64)
- **Scheduler**: manages WAITING → RUNNING → FINISHED lifecycle
  - Prefill: variable-length batched flash attention
  - Decode: CUDA-graph-captured single-token steps
- **ModelRunner**: orchestrates GPU execution
  - `prepare_prefill()` → slot_mapping, cu_seqlens
  - `prepare_decode()` → block_tables, context_lens
  - Triton kernel `store_kvcache` for zero-copy KV writes
- **BlockManager**: paged pool with reference counting
  - `allocate()` / `deallocate()` / `can_append()`
  - Hash-based prefix reuse across requests

### Slide 3: KVCOMM Engine Variants

| Feature | KVCOMMEngine | PagedKVCOMMEngine |
|---------|-------------|-------------------|
| Backend | HuggingFace DynamicCache | nano-vllm blocks |
| Storage | Full KV tensor copies | Block table references |
| Memory | O(anchors × seq_len × hidden) | O(blocks) shared pool |
| Write | Python tensor assignment | Triton store_kvcache |
| Read | Direct tensor slice | Gather from block pool |

### Slide 4: PagedKVCOMMEngine — Key Design

- **Shares the exact same `kv_cache` tensor and `BlockManager`** with nano-vllm's ModelRunner
  - True zero-copy: model writes KV → engine reads same blocks
- **PagedAnchorEntry**: lightweight anchor representation
  - `block_table`: list of block IDs (not tensor copies!)
  - `ph_key/value_embedding`: small CPU tensors for similarity matching
  - `agent_deltas`: per-agent delta stored as separate block refs
- **Block lifecycle**: reference counting + proactive eviction
  - `evict_proactive()` when free pool ratio < threshold
  - LRU-based eviction using `anchor_info` activation counts
- **Thread safety**: single `_lock` for atomic anchor mutations
  - All heavy computation (KV reads, delta, block alloc) done OUTSIDE lock

### Slide 5: PagedKVCOMMEngine — Core Operations

- **`read_kv_from_blocks(block_table, num_tokens)`**
  - Gather blocks → reshape → trim to actual tokens → transpose
  - Returns `[num_layers, num_kv_heads, num_tokens, head_dim]`
- **`write_kv_to_blocks(block_table, key, value, num_tokens)`**
  - Compute slot_mapping → call Triton `store_kvcache` per layer
  - Scatter KV into paged cache in-place
- **`set_anchor(agent_id, ph_id, message, real_blocks, base_blocks, ...)`**
  - Read real KV and base KV from blocks
  - Compute delta: `delta = real_KV - base_KV`
  - Allocate new blocks, write delta
  - Atomically store entry with delta block refs
- **`offset_kv_cache(agent_id, ph_id, base_blocks, anchor_list)`**
  - Compute similarity weights via L2 distance on embeddings
  - Weighted sum of historical deltas: `new_KV = base_KV + Σ(w_i × delta_i)`
  - Write result to freshly allocated blocks

### Slide 6: PagedLLMChat — The Integration Layer

- Drop-in replacement for `LLMChat`, registered as `'PagedLLMChat'`
- **Shared class-level resources** (all agent instances share):
  - `_shared_engine` (LLMEngine), `_paged_kv_engine` (PagedKVCOMMEngine)
  - `_shared_kv_cache_memory` (per-agent prefix block stores)
- **`_initialize_shared_resources()`**:
  - Creates LLMEngine → extracts `model_runner.kv_cache` and `scheduler.block_manager`
  - Passes them to PagedKVCOMMEngine constructor → **same memory, zero copy**
- **`_generate_tokens()`**: drives nano-vllm's scheduler loop
  - Creates `Sequence` with optional `prefilled_block_table` for KV reuse
  - Pins blocks before scheduler deallocates on EOS
  - Returns `(completion_tokens, ttft, prefill_latency, seq)`

### Slide 7: PagedLLMChat — KV Reuse Pipeline

- **Phase 1: Initialization (`prepare_prefix_kv_segments`)**
  - Prefill the base prompt template (no placeholder content)
  - Store prefix `block_table` per agent
  - Register base anchors via `register_base_anchor()`
- **Phase 2: KV Reuse Generation (`_prepare_kv_reuse_prefix_blocks`)**
  - For each placeholder span in token order:
    - **Path 1 — Current-Round Sharing (CRS)**: `user_question` only
      - `new_KV = base + (delta_upstream_current - cross_delta_historical)`
      - Reuses a peer agent's freshly computed KV with cross-agent correction
    - **Path 2 — KVCOMM Anchor Matching**: non-`user_question` placeholders
      - Calls `offset_kv_cache()` with historical anchor deltas
      - Weighted delta blending based on embedding similarity
    - **Path 3 — Fallback**: use base template blocks (ref-incremented)
  - Assemble combined block table: gap blocks + modified blocks + tail blocks
- **Phase 3: Generation**
  - Pass combined `cached_prefix_block_table` to `_generate_tokens()`
  - nano-vllm skips prefill for cached blocks → only prefills new tokens
  - After generation: `set_anchor()` stores delta for future rounds

### Slide 8: Data Flow — End-to-End Example

```
Round 1 (initialization):
  Agent A: prefill full prompt → store prefix blocks → generate → set_anchor
  Agent B: prefill full prompt → store prefix blocks → generate → set_anchor

Round 2+ (kv_reuse):
  Agent A:
    1. Build prompt with new user_question
    2. For user_question span: CRS from Agent B's delta
    3. For role/constraint spans: KVCOMM anchor matching (historical deltas)
    4. Assemble cached block table
    5. _generate_tokens(cached_prefix) → skip prefill for cached portion
    6. set_anchor() → store new delta for future rounds
  Agent B: (same, symmetric)
```

### Slide 9: Key Benefits of Paged Architecture

- **Memory Efficiency**: block references instead of tensor copies
  - Anchor storage: O(num_blocks) integers vs O(layers × heads × tokens × dim) floats
- **Zero-Copy Integration**: PagedKVCOMMEngine reads/writes the SAME kv_cache as model
  - No data movement between engine and KVCOMM
- **Dynamic Memory Management**: proactive eviction + reference counting
  - Anchors freed when pool pressure rises, blocks recycled
- **Reduced TTFT**: cached prefix blocks skip prefill entirely
  - Only new/modified tokens need forward pass
- **Scalability**: shared block pool across all agents
  - No per-agent memory reservation needed

---

## Part 3: English Speech Script

### Opening (Slide 1)

Good afternoon everyone. Today I'm going to present our work on PagedKVCOMM — a system that brings paged attention to multi-agent KV cache reuse.

The core problem we're solving is this: in a multi-agent LLM framework, multiple agents share a common prompt template — they have the same system prompt, the same role description, but different content in certain placeholder fields. When we run these agents, each one performs a full prefill of the entire prompt, which is extremely wasteful because most of the prompt is identical across agents.

KVCOMM's key insight is that the KV caches for these shared portions are similar but not identical — they differ because of attention's context dependency. So instead of recomputing from scratch, we store historical KV caches as "anchors" and approximate new ones using weighted delta blending.

### nano-vllm Engine (Slide 2)

Before diving into the KVCOMM integration, let me briefly introduce nano-vllm, the inference engine we build on.

nano-vllm is a minimalist vLLM-style inference engine. At the top level, we have an LLMEngine that orchestrates three components: a Scheduler, a ModelRunner, and a Tokenizer.

The Scheduler manages the lifecycle of each generation request — it moves sequences through three states: WAITING, RUNNING, and FINISHED. It handles block allocation through the BlockManager and supports preemption when GPU memory is tight.

The ModelRunner handles the actual GPU computation. For prefill, it uses flash attention with variable-length batching. For decode, it captures CUDA graphs for efficiency. Both phases use paged attention — KV cache is stored in fixed-size blocks, and a Triton kernel called `store_kvcache` writes KV data directly into these blocks with zero copies.

The BlockManager maintains a pool of KV cache blocks with reference counting. It supports prefix caching through semantic hash matching — if two requests share the same token prefix, they share the same physical KV blocks.

### KVCOMM Engine Comparison (Slide 3)

Now, the original KVCOMM used HuggingFace's DynamicCache for anchor storage. Each anchor stored a full copy of the KV tensors. This works fine for small-scale experiments, but the memory cost scales linearly with the number of anchors times sequence length times hidden dimension.

Our PagedKVCOMMEngine replaces this with block-table references. Instead of copying tensors, we just store a list of block IDs that point into the shared KV cache pool. This is a fundamental architectural shift — from O(n) tensor copies to O(1) integer references.

### PagedKVCOMMEngine Design (Slides 4-5)

The key design decision is that PagedKVCOMMEngine shares the exact same `kv_cache` tensor and `BlockManager` instance as nano-vllm's ModelRunner. When the model runs a forward pass and writes KV into blocks, our engine can immediately read those same blocks — true zero-copy.

Each anchor is stored as a `PagedAnchorEntry` with three components: a block table for the base KV, small CPU-resident embeddings for similarity matching, and per-agent delta block tables. The deltas capture how each agent's actual KV differs from the base template KV.

The core operations are: `read_kv_from_blocks` which gathers blocks and reconstructs continuous tensors; `write_kv_to_blocks` which scatters tensors back into blocks using Triton; `set_anchor` which computes and stores the delta between real and base KV; and `offset_kv_cache` which blends historical deltas weighted by embedding similarity.

For thread safety, all heavy computation — KV reads, delta computation, block allocation — happens outside the lock. Only the final atomic store of the anchor entry requires the lock. This minimizes contention in the multi-agent setting.

### PagedLLMChat Integration (Slides 6-7)

PagedLLMChat is the integration layer that ties everything together. It's a drop-in replacement for the original LLMChat, meaning you can switch between paged and non-paged backends just by changing a config flag.

During initialization, it creates the nano-vllm LLMEngine, then extracts the `kv_cache` and `block_manager` directly from the engine's ModelRunner and Scheduler. These are passed to the PagedKVCOMMEngine constructor — so we have one unified memory pool, not two separate ones.

The KV reuse pipeline has three phases. In Phase 1, during initialization, we prefill the base prompt template and store the resulting block table. We also register base anchors for each placeholder span.

In Phase 2, for each new generation request, we build a modified prefix by processing each placeholder span. For the `user_question` placeholder, we use Current-Round Sharing — we take a peer agent's freshly computed delta and apply a cross-agent correction learned from historical data. For other placeholders like role descriptions, we use KVCOMM anchor matching — finding similar historical anchors and blending their deltas.

In Phase 3, we pass the assembled cached block table to the generation function. nano-vllm recognizes the cached prefix and skips prefill for those blocks — only the new or modified tokens need a forward pass. After generation, we store the new delta as an anchor for future rounds.

### End-to-End Flow (Slide 8)

Let me walk through a concrete example. In Round 1, both agents do full prefill — there are no historical anchors yet. After generation, each stores its KV delta.

From Round 2 onward, when Agent A needs to generate with a new user question, it first checks if Agent B has already computed this question's KV — if so, it uses Current-Round Sharing with cross-agent correction. For the role and constraint placeholders, it looks up the most similar historical anchors and blends their deltas.

The result is a combined block table where most blocks are already filled with approximate KV — only a small fraction needs actual prefill computation.

### Benefits (Slide 9)

To summarize the key benefits: first, memory efficiency — we store block IDs instead of tensor copies, which is orders of magnitude smaller. Second, zero-copy integration — the KVCOMM engine and the inference engine share the same physical memory. Third, dynamic memory management with proactive eviction and reference counting. And most importantly, reduced Time-To-First-Token — cached prefix blocks skip prefill entirely, and the weighted delta approximation is accurate enough that generation quality is maintained.

Thank you. I'm happy to take questions.
