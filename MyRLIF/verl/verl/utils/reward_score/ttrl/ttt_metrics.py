from collections import Counter
from typing import List, Dict, Tuple, Optional
import re
import math
from difflib import SequenceMatcher
import numpy as np
import torch
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False
try:
    from transformers import AutoTokenizer, AutoModel
    _HAS_HF = True
except Exception:
    _HAS_HF = False

from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer


def test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    task="math", extra_info=None):
    
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"

    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)
    counter = Counter(model_answers)
    
    estimated_label, majority_count = counter.most_common(1)[0]
    
    hit_rate = 1.0 if auto_verify(task, [estimated_label], [ground_truth], extra_info=extra_info)[0][0] else 0.0
    majority_ratio = majority_count / len(solutions)
    # true_label_ratio = counter.get(ground_truth, 0) / len(solutions)

    rewards, _ = auto_verify(task, solutions, [estimated_label] * len(solutions), extra_info=extra_info)
    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    
    rewards_hit_rate = 0
    for reward, true_reward in zip(rewards, true_rewards):
        if reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(rewards)

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

    ttrl_metrics = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_ratio": majority_ratio,
        "ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        "majority_voting_reward": sum(rewards) / len(rewards),
        f"pass@{len(solutions)}": 1.0 if sum(true_rewards) >= 1 else 0.0,
    }
    return rewards, ttrl_metrics

def post_test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    pred_rewards: List,
    task="math", extra_info=None):
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(solutions) == len(pred_rewards), f"{len(solutions)} vs {len(pred_rewards)}"
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    # Compute true binary rewards
    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)

    # Robust comparison for continuous rewards
    pos = [r for r, t in zip(pred_rewards, true_rewards) if t == 1]
    neg = [r for r, t in zip(pred_rewards, true_rewards) if t == 0]

    # AUC-like metric: probability a random positive has higher score than a random negative
    def _pairwise_auc(pos_list, neg_list):
        if len(pos_list) == 0 or len(neg_list) == 0:
            return 0.0
        wins = 0.0
        total = 0.0
        for p in pos_list:
            for n in neg_list:
                total += 1.0
                if p > n:
                    wins += 1.0
                elif p == n:
                    wins += 0.5
        return wins / total if total > 0 else 0.0

    auc_pref = _pairwise_auc(pos, neg)

    # Mean rewards by class
    mean_pos = sum(pos) / len(pos) if len(pos) > 0 else 0.0
    mean_neg = sum(neg) / len(neg) if len(neg) > 0 else 0.0

    # Normalize reward accuracy via threshold at group mean (simple proxy)
    mean_all = sum(pred_rewards) / len(pred_rewards) if len(pred_rewards) > 0 else 0.0
    bin_pred = [1 if r >= mean_all else 0 for r in pred_rewards]
    rewards_hit_rate = sum(1 if bp == tr else 0 for bp, tr in zip(bin_pred, true_rewards)) / len(bin_pred)

    post_ttrl_metrics = {
        "post_reward_accuracy": rewards_hit_rate,
        "post_reward_auc_preference": auc_pref,
        "post_reward_mean_true": mean_pos,
        "post_reward_mean_false": mean_neg,
        "post_mean_train_accuracy": sum(true_rewards) / len(true_rewards),
        f"post_pass@{len(solutions)}": 1.0 if sum(true_rewards) > 0 else 0.0,
    }
    return post_ttrl_metrics


# --------------------
# Custom step-based reverse AUC rewards
# --------------------

_STEP_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:"
    r"Step\s*\d+|"           # English: Step 1:
    r"步骤\s*\d+|"            # Chinese: 步骤1：
    r"第\s*\d+\s*步|"        # Chinese: 第1步
    r"\d+[\.\):：、])\s+"   # 1.  2)  3:  4、
)


def _normalize_step_text(text: str) -> str:
    # Lowercase, remove excessive whitespace, and strip non-essential punctuation while
    # keeping common math symbols.
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff=+\-*/^()\[\]{}<>.,:; ]+", "", text)
    return text.strip()


def _split_into_steps(response: str) -> List[str]:
    if not response or not isinstance(response, str):
        return []
    matches = list(_STEP_PATTERN.finditer(response))
    if not matches:
        return []
    steps = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
        step = response[start:end].strip()
        if step:
            steps.append(step)
    return steps


def _extract_answer_step(response: str) -> str:
    """Extract final answer text, prioritizing LaTeX \boxed{...} content.

    Tries parser `extract_answer` first; falls back to regex for
    "\\boxed{...}" and "\\box{...}" if needed. Returns empty string
    when no final answer is detected.
    """
    if not response or not isinstance(response, str):
        return ""
    # Preferred: robust parser
    try:
        ans = extract_answer(response)
        if isinstance(ans, str) and ans.strip():
            return ans.strip()
    except Exception:
        pass
    # Fallback: simple regex for latex boxed content
    for pattern in [r"\\boxed\s*\{([^{}]+)\}", r"\\box\s*\{([^{}]+)\}"]:
        m = re.search(pattern, response)
        if m:
            content = m.group(1).strip()
            if content:
                return content
    return ""


def _cluster_steps(all_steps: List[str], sim_threshold: float = 0.8) -> Tuple[List[str], List[int], List[int]]:
    # Returns (representatives, counts, assignment_indices for each step in all_steps)
    representatives: List[str] = []
    counts: List[int] = []
    assignments: List[int] = []

    for step in all_steps:
        norm_step = _normalize_step_text(step)
        if not norm_step:
            # treat empty after normalization as its own unique cluster
            representatives.append(norm_step)
            counts.append(1)
            assignments.append(len(representatives) - 1)
            continue

        found = False
        for idx, rep in enumerate(representatives):
            # skip empty representative
            if not rep:
                continue
            sim = SequenceMatcher(None, norm_step, rep).ratio()
            if sim >= sim_threshold:
                counts[idx] += 1
                assignments.append(idx)
                found = True
                break
        if not found:
            representatives.append(norm_step)
            counts.append(1)
            assignments.append(len(representatives) - 1)
    return representatives, counts, assignments


_EMBED_MODEL_CACHE: Dict[str, object] = {}
_EMBED_TOKENIZER_CACHE: Dict[str, object] = {}


class TextEmbedder:
    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.backend = None
        self.model = None
        self.tokenizer = None
        if _HAS_ST:
            try:
                self.model = SentenceTransformer(model_name, device=self.device)
                self.backend = 'st'
            except Exception:
                self.model = None
        if self.model is None and _HAS_HF:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model = self.model.to(self.device)
                self.backend = 'hf'
            except Exception:
                self.model = None
                self.tokenizer = None
                self.backend = None

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        if len(texts) == 0:
            return np.zeros((0, 1), dtype=np.float32)
        if self.model is None:
            return None
        if self.backend == 'st':
            try:
                emb = self.model.encode(texts, normalize_embeddings=True)
                return np.asarray(emb, dtype=np.float32)
            except Exception:
                return None
        if self.backend == 'hf':
            try:
                inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                last = outputs.last_hidden_state  # [B, T, H]
                mask = inputs['attention_mask'].unsqueeze(-1)  # [B, T, 1]
                summed = (last * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1)
                mean = summed / denom  # [B, H]
                mean_norm = torch.nn.functional.normalize(mean, p=2, dim=1)
                return mean_norm.detach().cpu().numpy().astype(np.float32)
            except Exception:
                return None
        return None


def build_embedder(model_name: str, device: Optional[str] = None) -> Optional[TextEmbedder]:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
    elif device == 'cuda' and torch.cuda.device_count() == 0:
        device = 'cpu'
    embedder = TextEmbedder(model_name=model_name, device=device)
    if embedder.model is None:
        return None
    return embedder


def _embed_texts(texts: List[str], model_name: str, device: Optional[str] = None) -> Optional[np.ndarray]:
    if len(texts) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
    elif device == 'cuda' and torch.cuda.device_count() == 0:
        device = 'cpu'
    # Prefer sentence-transformers if available
    if _HAS_ST:
        if model_name not in _EMBED_MODEL_CACHE:
            _EMBED_MODEL_CACHE[model_name] = SentenceTransformer(model_name, device=device)
        model = _EMBED_MODEL_CACHE[model_name]
        # Enforce a safe maximum sequence length to avoid position embedding overflow
        try:
            if hasattr(model, "max_seq_length"):
                current_max = int(getattr(model, "max_seq_length", 512))
                # Bound to 512 as a safe default for BERT-like encoders
                model.max_seq_length = min(current_max if current_max > 0 else 512, 512)
        except Exception:
            pass
        try:
            emb = model.encode(texts, normalize_embeddings=True)
            return np.asarray(emb, dtype=np.float32)
        except Exception:
            pass  # fall through to HF transformers
    # Fallback to plain transformers
    if _HAS_HF:
        if model_name not in _EMBED_TOKENIZER_CACHE:
            _EMBED_TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
        if model_name not in _EMBED_MODEL_CACHE:
            _EMBED_MODEL_CACHE[model_name] = AutoModel.from_pretrained(model_name)
        tok = _EMBED_TOKENIZER_CACHE[model_name]
        mdl = _EMBED_MODEL_CACHE[model_name]
        mdl = mdl.to(device)
        # Determine and enforce a safe max_length to prevent overflow vs. model position embeddings
        try:
            max_len = int(getattr(mdl.config, "max_position_embeddings", 512))
        except Exception:
            max_len = int(getattr(tok, "model_max_length", 512)) if hasattr(tok, "model_max_length") else 512
        # Guard against sentinel or unreasonable values
        if max_len is None or max_len <= 0 or max_len > 4096 or max_len >= 100000:
            max_len = 512
        inputs = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = mdl(**inputs)
        last = outputs.last_hidden_state  # [B, T, H]
        mask = inputs['attention_mask'].unsqueeze(-1)  # [B, T, 1]
        summed = (last * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        mean = summed / denom  # [B, H]
        # L2 normalize
        mean_norm = torch.nn.functional.normalize(mean, p=2, dim=1)
        return mean_norm.detach().cpu().numpy().astype(np.float32)
    return None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    num = float(np.dot(a, b.T))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0:
        return 0.0
    return num / den


def _cluster_steps_embeddings(
    all_steps: List[str],
    model_name: str,
    device: Optional[str],
    sim_threshold: float,
    embedder: Optional[TextEmbedder] = None,
) -> Tuple[List[str], List[int], List[int]]:
    reps: List[str] = []
    counts: List[int] = []
    assignments: List[int] = []
    centroids: List[np.ndarray] = []
    texts = [_normalize_step_text(s) for s in all_steps]
    emb: Optional[np.ndarray] = None
    if embedder is not None:
        emb = embedder.encode(texts)
    if emb is None:
        emb = _embed_texts(texts, model_name=model_name, device=device)
    if emb is None:
        # Fallback to simple text similarity if embedding unavailable
        return _cluster_steps(all_steps, sim_threshold=sim_threshold)
    for i, vec in enumerate(emb):
        t = texts[i]
        if not t:
            reps.append(t)
            counts.append(1)
            assignments.append(len(reps) - 1)
            centroids.append(vec)
            continue
        found = False
        for idx, c in enumerate(centroids):
            if reps[idx] == "":
                continue
            sim = _cosine_sim(vec, c)
            if sim >= sim_threshold:
                # update centroid as running mean
                new_count = counts[idx] + 1
                centroids[idx] = (c * counts[idx] + vec) / new_count
                counts[idx] = new_count
                assignments.append(idx)
                found = True
                break
        if not found:
            reps.append(t)
            counts.append(1)
            assignments.append(len(reps) - 1)
            centroids.append(vec)
    return reps, counts, assignments


def reverse_auc_rewards_for_group(
    solutions: List[str],
    penalty_reward: float = -1,
    sim_threshold: float = 0.8,
    cluster_backend: str = "embedding",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embed_device: Optional[str] = None,
    embedder: Optional[TextEmbedder] = None,
) -> Tuple[List[float], Dict[str, float]]:
    """Compute step-based reverse AUC rewards for a group of rollout solutions.

    - Split each solution into enumerated steps (1., 2., ... / Step x: / 第x步 / 步骤x：).
    - Additionally, prepend the extracted final answer (e.g., LaTeX \boxed{...}) as step 1 if available.
    - Cluster all steps across the group by semantic similarity (SequenceMatcher or embeddings).
    - For each solution, get the probability of each step's cluster as count(cluster)/total_steps.
    - Compute a reverse AUC-style reward that prefers higher probabilities at later steps using a weighted geometric mean:
      reward = exp( sum_i [w_i * log(max(p_i, eps))] / sum_i w_i ), w_i = i.
      This avoids zero-collapse via eps and emphasizes later steps.
    - If a solution has no detectable steps (or gibberish by this heuristic), assign a unified penalty.
    """
    # Parse steps per solution, with answer step prepended when available
    steps_per_output: List[List[str]] = []
    for s in solutions:
        s = s or ""
        ans_step = _extract_answer_step(s)
        steps = _split_into_steps(s)
        if ans_step:
            steps = [ans_step] + steps
        steps_per_output.append(steps)
    all_steps: List[str] = []
    for steps in steps_per_output:
        all_steps.extend(steps)

    if len(all_steps) == 0:
        rewards = [penalty_reward for _ in solutions]
        metrics = {
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "invalid_ratio": 1.0,
        }
        return rewards, metrics

    # Cluster steps across the group
    if cluster_backend == "embedding":
        reps, counts, assignments = _cluster_steps_embeddings(
            all_steps,
            model_name=embedding_model_name,
            device=embed_device,
            sim_threshold=sim_threshold,
            embedder=embedder,
        )
    else:
        reps, counts, assignments = _cluster_steps(all_steps, sim_threshold=sim_threshold)
    total_steps = len(all_steps)
    cluster_count_map = {idx: c for idx, c in enumerate(counts)}

    # Build mapping from global step index to its cluster assignment
    # global_steps_offsets[i] marks starting index in all_steps for solution i
    global_steps_offsets = []
    offset = 0
    for steps in steps_per_output:
        global_steps_offsets.append(offset)
        offset += len(steps)

    rewards: List[float] = []
    invalid = 0
    for i, steps in enumerate(steps_per_output):
        n = len(steps)
        if n == 0:
            rewards.append(penalty_reward)
            invalid += 1
            continue
        # Weighted geometric mean across step cluster probabilities
        # weights w_i = i (later steps get larger weight)
        W = n * (n + 1) / 2.0  # sum of 1..n
        start = global_steps_offsets[i]
        sum_log = 0.0
        eps = 1e-9
        for j in range(n):
            global_idx = start + j
            cluster_idx = assignments[global_idx]
            p = cluster_count_map.get(cluster_idx, 0) / float(total_steps)
            weight = (j + 1)
            # guard against p == 0 with smoothing
            sum_log += weight * math.log(p if p > eps else eps)
        reward = math.exp(sum_log / W)
        rewards.append(reward)

    # Complementary metrics: arithmetic and geometric mean across solutions
    if rewards:
        mean_reward = float(sum(rewards)) / float(len(rewards))
        mean_reward_gmean = math.exp(sum(math.log(r if r > 1e-12 else 1e-12) for r in rewards) / float(len(rewards)))
    else:
        mean_reward = 0.0
        mean_reward_gmean = 0.0

    metrics = {
        "mean_reward": mean_reward,
        "mean_reward_gmean": mean_reward_gmean,
        "invalid_ratio": invalid / float(len(solutions)) if solutions else 0.0,
        "cluster_count": float(len(reps)),
    }
    return rewards, metrics