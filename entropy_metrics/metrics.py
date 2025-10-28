import torch
import math


# 说明：
# 本文件实现四种与 RLIF 相关的“熵/确定性指标”计算函数。
# 为了数值稳定性，尽量使用 log_softmax / softmax 的形式。


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    令牌级熵 H(p_t) 的逐步序列（按时间步 t）。
    输入：
      - logits: [T, V] 或 [B, T, V]，其中 T 为时间步长度，V 为词汇表大小。
    输出：
      - ent: [T] 或 [B, T] 的熵序列，单位为 nats（自然对数）。
    """
    # 统一为 [*, T, V] 的形状，便于批量处理
    orig_shape = logits.shape
    if logits.dim() == 2:
        x = logits.unsqueeze(0)  # [1, T, V]
    elif logits.dim() == 3:
        x = logits
    else:
        raise ValueError("logits 维度必须是 [T, V] 或 [B, T, V]")

    # p = softmax(logits)
    log_p = torch.log_softmax(x, dim=-1)
    p = torch.exp(log_p)

    # 熵 H(p) = -sum p * log p
    ent = -(p * log_p).sum(dim=-1)  # [B, T]

    if len(orig_shape) == 2:
        return ent.squeeze(0)  # [T]
    return ent  # [B, T]


def self_certainty_kl_uniform(logits: torch.Tensor) -> torch.Tensor:
    """
    自我确定性：D_KL(U || p_t) 的逐步序列（按时间步 t）。
    - U 为均匀分布（1/V）。
    - p_t 为模型在时间步 t 的下一令牌分布。

    D_KL(U || p) = sum_i (1/V) * [log(1/V) - log p_i] = log(1/V) - mean_i[log p_i]
    其中 log p_i 使用 log_softmax 计算以保证稳定性。
    输入：
      - logits: [T, V] 或 [B, T, V]
    输出：
      - kl: [T] 或 [B, T]
    """
    orig_shape = logits.shape
    if logits.dim() == 2:
        x = logits.unsqueeze(0)  # [1, T, V]
    elif logits.dim() == 3:
        x = logits
    else:
        raise ValueError("logits 维度必须是 [T, V] 或 [B, T, V]")

    V = x.size(-1)
    log_p = torch.log_softmax(x, dim=-1)  # [B, T, V]
    mean_log_p = log_p.mean(dim=-1)       # [B, T]
    kl = math.log(1.0 / V) - mean_log_p   # [B, T]

    if len(orig_shape) == 2:
        return kl.squeeze(0)  # [T]
    return kl  # [B, T]


def trajectory_avg_logprob(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """
    轨迹级：平均对数概率 (1/|y|) * sum_t log p(y_t | x, y_{<t})。
    输入：
      - logits: [T, V] 或 [B, T, V]，为每一步的下一令牌的logits。
      - target_ids: [T] 或 [B, T]，为实际生成的目标令牌ID序列（与时间步对齐）。
    输出：
      - avg_logprob: 标量（若输入无批次则为标量张量；有批次则 [B]）。
    """
    # 统一形状
    if logits.dim() == 2:
        x = logits.unsqueeze(0)          # [1, T, V]
        y = target_ids.unsqueeze(0)      # [1, T]
    elif logits.dim() == 3:
        x = logits                        # [B, T, V]
        y = target_ids                    # [B, T]
    else:
        raise ValueError("logits 维度必须是 [T, V] 或 [B, T, V]")

    # 取目标令牌的 log p
    log_p = torch.log_softmax(x, dim=-1)  # [B, T, V]
    # 使用gather收集每步目标令牌的log概率
    tgt_logp = log_p.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)  # [B, T]

    # 平均对数概率
    avg_logp = tgt_logp.mean(dim=-1)  # [B]

    if logits.dim() == 2:
        return avg_logp.squeeze(0)     # 标量
    return avg_logp                    # [B]


def policy_entropy_dataset(entropy_series_list: list) -> float:
    """
    策略熵（数据集平均）：对所有样本、所有时间步的熵做平均。
    输入：
      - entropy_series_list: 列表，元素为每个样本的熵序列张量 [T] 或 [B, T]
    输出：
      - 标量 float，表示跨数据集的平均策略熵。
    """
    total = 0.0
    count = 0
    for ent in entropy_series_list:
        # 支持 [T] 或 [B, T]
        if ent.dim() == 1:
            total += float(ent.sum().item())
            count += int(ent.numel())
        elif ent.dim() == 2:
            total += float(ent.sum().item())
            count += int(ent.numel())
        else:
            raise ValueError("熵序列维度必须是 [T] 或 [B, T]")
    if count == 0:
        return float('nan')
    return total / count