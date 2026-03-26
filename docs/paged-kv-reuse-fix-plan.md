# Paged KV Reuse Fix Plan (MVP)

## Context

当前问题是：paged 路径虽然调用了 anchor/offset 逻辑，但没有稳定进入真正的生成路径，导致 KV reuse 对 TTFT 改善不明显，且日志层面难以证明“复用真的生效”。

## What Is Wrong (Code-Level)

### 1) Reuse 时序错误：先生成，再 offset

- 生成发生在 [KVCOMM/llm/paged_llm_chat.py](KVCOMM/llm/paged_llm_chat.py#L683)
- kv_reuse 分支在生成后才调用 offset，位置见 [KVCOMM/llm/paged_llm_chat.py](KVCOMM/llm/paged_llm_chat.py#L766)

影响：
- 当前请求的输出已经产生，offset 再正确也无法影响本轮生成。

### 2) offset 返回的新块未被消费

- offset 调用见 [KVCOMM/llm/paged_llm_chat.py](KVCOMM/llm/paged_llm_chat.py#L788)
- 返回值 new_ph_blocks/new_pf_blocks 没有进入后续 scheduler/model_runner 输入。

影响：
- 出现“计算了复用结果但逻辑上被丢弃”的行为。

### 3) 调度层复用入口未显式接入

- Sequence 初始化将 num_cached_tokens 置 0：见 [nanovllm/engine/sequence.py](nanovllm/engine/sequence.py#L25)
- Scheduler 仅依据 num_cached_tokens 统计 prefill 工作量：见 [nanovllm/engine/scheduler.py](nanovllm/engine/scheduler.py#L35)
- ModelRunner prefill 真正按 num_cached_tokens 截断输入：见 [nanovllm/engine/model_runner.py](nanovllm/engine/model_runner.py#L148)

影响：
- 若调用方不能稳定设置“哪些 token 已缓存”，就无法稳定跳过 prefill。

### 4) Placeholder token 区间直接切 block，存在错位风险

- dense_prefill 的切块位置见 [KVCOMM/llm/paged_llm_chat.py](KVCOMM/llm/paged_llm_chat.py#L704)
- kv_reuse 的切块位置见 [KVCOMM/llm/paged_llm_chat.py](KVCOMM/llm/paged_llm_chat.py#L773)

影响：
- 当 token 到 block 映射不再是理想线性关系时，可能切到错误块，进而污染 set_anchor/offset 语义。

### 5) Engine 语义与调用方语义脱节

- offset_kv_cache 返回新块表语义在 [KVCOMM/llm/paged_kvcomm_engine.py](KVCOMM/llm/paged_kvcomm_engine.py#L321)
- 返回语句见 [KVCOMM/llm/paged_kvcomm_engine.py](KVCOMM/llm/paged_kvcomm_engine.py#L446)

影响：
- Engine 端实现了“产出可复用块”的能力，但调用方未把结果注入生成链路。

## MVP Fix Scope

1. 修正 kv_reuse 时序：将 offset 与块替换接入到生成前/至少在 decode 前可生效的阶段。  
2. 修正 offset 结果消费：返回块必须被 scheduler/model_runner 使用。  
3. 增加显式可观测指标：每次请求输出 anchor 命中数、复用块数、跳过 prefill token 数。  
4. 加入显式降级日志：条件不满足时回退 dense_prefill，并记录具体原因。  

## Acceptance Criteria

1. kv_reuse 在同样本集上的 TTFT 中位数与均值优于 dense_prefill。  
2. 日志中可看到每次请求的缓存命中/跳过 prefill 计数。  
3. 小样本任务输出质量与 dense_prefill 相比无明显退化。  

## Suggested Implementation Order

1. 先补日志与计数器（低风险，先建立可观测性）。  
2. 再改 kv_reuse 时序和 offset 结果接入（核心逻辑修复）。  
3. 最后做 placeholder 切块稳健化（减少错位导致的隐性失败）。

## Implemented Delta (Current Branch)

1. 已新增可观测字段并写入 metadata/Latency：`num_cached_tokens`、`anchor_candidates`、`offset_calls`、`offset_effective`。  
2. 已统一 message 键策略，避免 set/get 键不一致导致的静默 miss。  
3. 已将 kv_reuse 的 offset 计算前置到生成前，构造预注入的 cached prefix block_table。  
4. 已在 BlockManager 增加“已有前缀块时仅分配 tail 块”的分配行为，支持 pre-injected prefix reuse。  

## Remaining Work

1. 多 placeholder 的稳健拼接策略（当前实现保守回退）。  
2. token 区间与 block 区间映射稳健化，降低错位概率。  
3. 跑 benchmark 与小样本回归，形成 TTFT 对比数据。
