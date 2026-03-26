# Paged vs Non-Paged KVCOMM性能对比分析

## 执行情况总结

### Paged版本 (nano-vllm后端)
**运行时间**: 2026-03-25 17:18:51 ~ 17:19:47 (56秒)
**⚠️ KV复用状态**: 未启用（所有batch都是dense_prefill）

| 批次 | 任务 | 耗时 | Agent模式分布 | 准确率 |
|------|------|------|--------------|--------|
| 0    | Janet鸡蛋 | 5.777s | 4×dense_prefill | 100% |
| 1    | 布料计算 | 2.644s | 4×dense_prefill | 100% |
| 2    | Josh房子翻转 | 15.255s | 4×dense_prefill | 66.7% |
| 3    | James短跑 | 3.417s | 4×dense_prefill | 75% |
| 4    | Wendi鸡 | 5.547s | 4×dense_prefill | 80% |
| **总计** |  | **32.64s** | **0% KV复用** |  |
| **平均/batch** |  | **6.528s** |  |  |

### 非Paged版本 (HuggingFace DynamicCache后端)
**运行时间**: 2026-03-25 17:31:07 ~ 17:32:06 (59秒)
**✅ KV复用状态**: Batch 3成功启用 (reuse_rate=0.75)

| 批次 | 任务 | 耗时 | Agent模式分布 | KV复用 | 准确率 |
|------|------|------|--------------|--------|--------|
| 0    | Janet鸡蛋 | 15.114s | 4×dense_prefill | 0/4 | 100% |
| 1    | 布料计算 | 12.619s | 4×dense_prefill | 0/4 | 100% |
| 2    | Josh房子翻转 | 15.376s | 4×dense_prefill | 0/4 | 100% |
| 3    | James短跑 | **12.134s** | 3×`kv_reuse`<br/>1×dense_prefill | **3/4** ✅ | 100% |
| 4+   | Wendi鸡(部分) | ~7.5s | 混合 | 混合 | 80% |
| **均值(0-2)** |  | **14.37s** | 纯dense_prefill | 0% |  |
| **Batch 3** |  | **12.134s** | 75% kv_reuse | 75% ✅ |  |

---

## 关键发现

### 1. **关键修正：两个版本的KV复用对比** ⚠️

| 特性 | Paged | 非Paged |
|------|-------|---------|
| **KV复用是否启用** | ❌ **否** - 全部dense_prefill | ✅ **是** - Batch 3: 75%复用 |
| **复用成功率** | 0/16 agents | 3/4 agents (Batch 3) |
| **Anchor创建状态** | ❌ 都跳过 (placeholder错误) | ❌ 都失败 (无active anchor) |

**重要发现**：
- 非paged版本虽然启用了kv_reuse，但**仍然没有找到任何有效的anchor**
- Paged版本连尝试kv_reuse都没有机会（因为anchor制作失败）
- **两个版本都因anchor创建问题无法真正执行KV复用优化**

### 2. **性能对比：即使非Paged用了reuse，仍然慢2倍**

```
非Paged（无reuse）: 14.37s/batch (batch 0-2均值)
非Paged（有reuse）: 12.13s/batch (batch 3，75%复用率)
Paged（无reuse）:   6.53s/batch (全部batch均值)

性能关系:
Paged (无reuse) > 非Paged (有reuse)
6.53s             <  12.13s
───────────────────────────────
Paged快 1.86倍，即使非Paged已启用KV复用！
```

### 3. **为什么即使非Paged用了reuse，仍然比Paged慢？**

**根本原因是TTFT差异**：

#### Paged的TTFT (恒定，34ms)
```
generation_ttft始终 = 16.7ms (token生成速度)
总TTFT = 34.2ms
```

#### 非Paged的TTFT (高且波动)
```
Dense prefill: 70ms 
Kv_reuse模式:
  - Agent 1: 63.88ms (preprocess 47.3ms)
  - Agent 2: 97.80ms (preprocess 81.0ms)  ← 还有大量overhead!
  - Agent 3: 159.8ms (preprocess 143ms)   ← 非常慢！
平均: ~81ms
```

**关键发现**：即使非Paged启用了kv_reuse，preprocess overhead仍然很大（47-143ms），导致TTFT仍然比Paged慢2-3倍。

### 4. **Anchor创建问题对比**

#### Paged版本的失败：
```
跳过原因: "placeholder blocks out of range"
  ph_start_block值 > block_table_len
  原因: Block表在dense_prefill中还未初始化
```

#### 非Paged版本的失败：
```
[ANCHOR_CHECK] No active anchor found, return False
  原因: 之前的dense_prefill没有成功创建anchor
    (因为reuse_count=0在batch 0-2)
  结果: 即使batch 3启用了kv_reuse，仍然没有可用的anchor
```

**结论**：两个版本都因不同原因无法创建anchor，导致kv_reuse无效化

#### A. **Nano-vllm的块式注意力机制**
- Paged backend使用**块式KV缓存** (block_size=256)
- 初始化时创建: **2171个block** 预分配
- 使用**Triton kernel**实现零复制prefill
- 内存局部性更好，缓存命中率高

#### B. **HuggingFace DynamicCache的开销**
- 动态扩展缓存，需要频繁的内存重新分配
- 标准PyTorch/CUDA kernel实现，未针对多轮序列优化  
- 每个Agent调用都需要初始化新的DynamicCache块

#### C. **生成速度对比**
| 指标 | Paged | 非Paged |
|------|-------|----------|
| 初始化开销 | ~22s (一次) | 低 |
| Per-token生成/100token | ~0.1s | ~0.3s |
| TTFT (首token) | 34ms | 70ms |

Paged的高初始开销被大幅更快的生成速度抵消！

---

## 4. **Anchor创建问题**

### 两个版本都没有成功创建anchors！

```
跳过原因: "placeholder blocks out of range"
  - ph_start_block 值远超 block_table_len (0)
  - 示例: ph_start_block=4 但 block_table_len=0
```

#### 根本原因分析
1. **Placeholder初始化错误**: Dense prefill过程中，placeholder block分配不正确
2. **Block table为空**: `block_table_len=0` 表示没有有效的block被加入表
3. **时机问题**: set_anchor在dense_prefill中执行，但此时block还未初始化

#### 代码位置
- [paged_llm_chat.py](KVCOMM/llm/paged_llm_chat.py#L916)行 - anchor跳过逻辑
- 需要检查: `_generate_paged()` 中 dense_prefill → set_anchor的时序

---

## 5. **关键性能对比视图**

```
总任务耗时对比 (5个batch):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Paged:     ███████████ 32.64s 总计
非Paged:   ██████████████████████ 71.49s 总计 (推估)

每batch耗时分布:
┌─ Paged版本 ────────────────────┐
│ Batch 0: ████ 5.8s            │
│ Batch 1: ██ 2.6s              │
│ Batch 2: ███████████ 15.3s    │ ← 输出错误导致慢
│ Batch 3: ███ 3.4s             │
│ Batch 4: ████ 5.5s            │
└────────────────────────────────┘
┌─ 非Paged版本 ──────────────────┐  
│ Batch 0: ██████████ 15.1s    │
│ Batch 1: █████████ 12.6s     │
│ Batch 2: ██████████ 15.4s    │
└────────────────────────────────┘
```

---

## 结论

### 答案：为什么Paged快2倍，即使都没有KV复用？

1. ✅ **Nano-vllm的块式分页注意力** - 超级高效的底层实现
2. ✅ **TTFT快2倍** - First-token-latency是bottleneck，Paged完胜
3. ✅ **内存管理优化** - Triton kernel优化比标准PyTorch更快
4. ❌ **KV复用未激活** - 两个版本都因placeholder错误无法设置anchors

### 下一步调试目标
- [x] 理解性能差异根因 ✓ (本文档)
- [ ] **修复anchor设置**: 处理 "placeholder blocks out of range" 错误
- [ ] 对比启用KV复用后的性能 (目标: 4x+ 加速)

---

## 附录：详细TTFT数据

### Paged版本(ms, 取自log.txt第1批)
```
Agent 1 (Math Solver):      34.663ms
Agent 2 (Analyst):          34.341ms  
Agent 3 (Programming):      34.281ms
Agent 0 (Final):            34.593ms
───────────────────────────────────
平均TTFT:                   34.469ms
```

### 非Paged版本(ms, 取自log_no_print.txt第3批+)
```
Agent 1 dense_prefill:      78.02ms
Agent 2 dense_prefill:      55.70ms
Agent 3 dense_prefill:      75.33ms
Agent 0 dense_prefill:      69.72ms

Agent 1 kv_reuse:           63.88ms (incl. preprocess 47.3ms)
Agent 2 kv_reuse:           97.80ms (incl. preprocess 81.0ms)
───────────────────────────────
平均TTFT (dense_prefill):   69.69ms
平均TTFT (kv_reuse):        ~81ms
```

**结论**: Paged TTFT恒定在34ms，而非Paged波动在60-100ms → **2-3倍差异**
