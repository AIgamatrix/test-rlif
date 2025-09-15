# 训练框架结果对比

基于 test_res_qwen 目录下四个数据集的测试结果对比分析。

## 总体结果对比表

| 数据集 | 题目数 | Origin/Qwen | EMPO | Intuitor | TTRL |
|--------|--------|-------------|------|----------|------|
| **AIME** | 90 | 7 (7.78%) | 5 (5.56%) | **8 (8.89%)** | **9 (10.0%)** |
| **AMC12** | 83 | 17 (20.48%) | 22 (26.51%) | **33 (39.76%)** | 24 (28.92%) |
| **GPQA** | 448 | **132 (29.46%)** | 118 (26.34%) | 112 (25.0%) | 113 (25.22%) |
| **Minerva** | 272 | **33 (12.13%)** | 30 (11.03%) | 29 (10.66%) | 21 (7.72%) |

*注：粗体表示该数据集上的最佳结果*

## 相对于原始结果的提升/下降（百分点）

| 数据集 | EMPO | Intuitor | TTRL |
|--------|------|----------|------|
| **AIME** | -2.22 pp | **+1.11 pp** | **+2.22 pp** |
| **AMC12** | **+6.03 pp** | **+19.28 pp** | **+8.44 pp** |
| **GPQA** | -3.12 pp | -4.46 pp | -4.24 pp |
| **Minerva** | -1.10 pp | -1.47 pp | -4.41 pp |

*注：pp = percentage points（百分点）*

## 主要发现

### 各框架表现特点：

1. **Intuitor**：
   - 在 AMC12 上表现最突出（39.8%，提升 19.28 pp）
   - 在 AIME 上也有小幅提升（+1.11 pp）
   - 在 GPQA 和 Minerva 上略有下降

2. **TTRL**：
   - 在 AIME 上表现最佳（10.0%，提升 2.22 pp）
   - 在 AMC12 上也有显著提升（+8.94 pp）
   - 在 GPQA 和 Minerva 上有所下降，特别是 Minerva（-4.41 pp）

3. **EMPO**：
   - 在 AMC12 上有中等程度提升（+6.01 pp）
   - 在其他数据集上均有小幅下降

4. **Origin/Qwen**：
   - 在 GPQA 和 Minerva 上仍保持最佳性能
   - 在数学推理任务（AIME、AMC12）上被训练框架超越

### 数据集特点：

- **数学推理任务**（AIME、AMC12）：训练框架普遍有提升
- **科学问答任务**（GPQA、Minerva）：原始模型表现更好

## 数据来源

所有结果基于以下文件的 `total_questions`、`correct_count` 和 `accuracy` 字段：

- Origin: `/home/llama/test-rlif/test_res_qwen/origin/qwen_*_test_results_summary.json`
- EMPO: `/home/llama/test-rlif/test_res_qwen/EMPO/*_results_summary.json`
- Intuitor: `/home/llama/test-rlif/test_res_qwen/Intuitor/*_results_summary.json`
- TTRL: `/home/llama/test-rlif/test_res_qwen/TTRL/*_results_summary.json`