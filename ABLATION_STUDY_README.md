# Flash Attention + Paged KV Cache Ablation Study

## 概述

这个 ablation study 设计用来逐步评估以下两个优化的独立影响和组合效果：
1. **Flash Attention 2** - 高效的注意力计算
2. **Paged KV Cache** - 块级 KV 缓存管理

## 实验设计

### 三个实验组

| ID | 后端 | Flash Attn | Paged KV | 目的 |
|---|---|:---:|:---:|---|
| **1** | gpt_chat (HF) | ❌ | ❌ | **基线** - 原始 HuggingFace 实现 |
| **2** | gpt_chat (HF) | ✅ | ❌ | **Flash Attention 单独影响** |
| **3** | nano-vllm | ✅ | ✅ | **联合效果** - FA + Paged |

### 性能指标

每个实验都会记录以下指标：

- **Accuracy**: 任务准确率
- **TTFT (Time To First Token)**: 首 token 延迟（prefill 性能）
- **Throughput**: 吞吐量
- **Memory Usage**: 内存占用
- **Latency Distribution**: 延迟分布统计

## 使用方式

### 方式 1：运行完整 ablation 脚本

```bash
cd /home/users/yw641/workspace/mlsys/PagedKVCOMM
sbatch KVCOMM/run_ablation.sh
```

或者直接运行（不使用 SLURM）：

```bash
bash KVCOMM/run_ablation.sh
```

脚本会依次运行三个实验，每个实验都有 10 秒的冷却时间。

### 方式 2：单独运行各个实验

**实验 1：基线（无 Flash Attention）**
```bash
export KVCOMM_PAGED=0

python -m KVCOMM.experiments.run_gsm8k \
  --llm_name meta-llama/Llama-3.1-8B-Instruct \
  --execution_mode allow_kv_reuse \
  --output_dir KVCOMM/result/ablation_1_baseline
```

**实验 2：仅 Flash Attention**
```bash
export KVCOMM_PAGED=0

python -m KVCOMM.experiments.run_gsm8k \
  --llm_name meta-llama/Llama-3.1-8B-Instruct \
  --execution_mode allow_kv_reuse \
  --output_dir KVCOMM/result/ablation_2_with_fa \
  --use-flash-attention
```

**实验 3：Paged + Flash Attention**
```bash
export KVCOMM_PAGED=1

python -m KVCOMM.experiments.run_gsm8k \
  --llm_name meta-llama/Llama-3.1-8B-Instruct \
  --execution_mode allow_kv_reuse \
  --output_dir KVCOMM/result/ablation_3_paged_fa
```

## 分析结果

运行完成后，使用分析脚本查看对比结果：

```bash
python scripts/analyze_ablation.py
```

这会生成一个对比表格，展示：
- 各实验的准确率
- 平均 TTFT 和加速比
- 关键洞见

### 示例输出

```
================================================================================
KVCOMM Ablation Study: Flash Attention + Paged KV Cache
================================================================================

Analyzing: Baseline (gpt_chat, no FA)
  ✓ Tests: 1319
  ✓ Solved: 842/1319
  ✓ Accuracy: 0.6381
  ✓ Avg TTFT: 0.0354s

Analyzing: With Flash Attention
  ✓ Tests: 1319
  ✓ Solved: 842/1319
  ✓ Accuracy: 0.6381
  ✓ Avg TTFT: 0.0312s

Analyzing: Paged (FA + Paged KV)
  ✓ Tests: 1319
  ✓ Solved: 842/1319
  ✓ Accuracy: 0.6381
  ✓ Avg TTFT: 0.0289s

================================================================================
COMPARISON TABLE
================================================================================

Metric                         Baseline             With FA              Paged
-------------------------------------------------------------------------------------------
Accuracy                       0.6381               0.6381               0.6381
Avg TTFT (s)                   0.0354               0.0312               0.0289
Speedup vs Baseline            1.00x                1.13x                1.22x
```

## 预期结果

### Flash Attention (Exp 2 vs Exp 1)
- ✅ 首 Token 延迟减少 5-15%
- ✅ 吞吐量提升 10-20%
- ✅ 内存使用减少 10-20%
- ✅ **准确率不变**（Flash Attention 不影响准确度）

### Paged KV Cache (Exp 3 vs Exp 2)
- ✅ 首 Token 延迟再减少 5-10%
- ✅ 内存用量进一步降低
- ✅ **准确率不变**

### 组合效果 (Exp 3 vs Exp 1)
- 预期总体加速比：**1.15-1.30x**
- 所有准确率保持一致

## 关键对比查询

### 问题 1: Flash Attention 能改进多少？
```
Speedup = Baseline_TTFT / WithFA_TTFT
```

### 问题 2: Paged 在 Flash Attention 基础上还能改进？
```
Additional_Speedup = WithFA_TTFT / Paged_TTFT
```

### 问题 3: 总体改进是多少？
```
Total_Speedup = Baseline_TTFT / Paged_TTFT
```

## 注意事项

1. **模型初始化**：第一次运行时会下载模型，可能需要较长时间
2. **GPU 内存**：确保有足够的 GPU 内存（建议 20GB+）
3. **公平对比**：
   - 每个实验之间有 10 秒的冷却间隔
   - 所有实验使用相同的数据集（前 424+ 条记录）
   - 所有实验使用相同的批处理大小和参数

4. **日志文件**：
   - 每个实验的详细日志保存在 `KVCOMM/logs/`
   - 结果 JSON 文件保存在 `KVCOMM/result/ablation_*`

## 文件位置

```
KVCOMM/
├── run_ablation.sh              # 完整 ablation 脚本
├── result/
│   ├── ablation_1_baseline/     # 实验 1 结果
│   ├── ablation_2_with_fa/      # 实验 2 结果
│   └── ablation_3_paged_fa/     # 实验 3 结果
└── logs/
    ├── ablation_1_baseline.log
    ├── ablation_2_with_fa.log
    └── ablation_3_paged_fa.log

scripts/
└── analyze_ablation.py          # 分析脚本
```

## 扩展实验

### 可选：多数据集测试

修改 `run_ablation.sh` 来测试其他数据集：

```bash
# MMLU
$PYTHON_BIN -m KVCOMM.experiments.run_mmlu \
  --llm_name "$LLAMA_PATH" \
  --execution_mode allow_kv_reuse \
  --output_dir KVCOMM/result/ablation_mmlu \
  [--use-flash-attention]

# HumanEval
$PYTHON_BIN -m KVCOMM.experiments.run_humaneval \
  --llm_name "$LLAMA_PATH" \
  --execution_mode allow_kv_reuse \
  --output_dir KVCOMM/result/ablation_humaneval \
  [--use-flash-attention]
```

## 故障排除

### 问题：模型加载失败
```
Solution: 检查 HF_HOME 和模型路径是否正确
export HF_HOME=/path/to/hf_cache
```

### 问题：CUDA out of memory
```
Solution: 增加批处理间隔或使用更小的模型
或设置: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 问题：时间太长
```
Solution: 修改脚本，只运行前 N 条记录（编辑 run_gsm8k.py 的 num_batches）
```

## 相关论文和参考

- Flash Attention: https://arxiv.org/abs/2205.14135
- Paged Attention (vLLM): https://arxiv.org/abs/2309.06180
- KVCOMM: [项目特定论文]
