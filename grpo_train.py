"""从 Instruct 模型出发的 GRPO 训练循环，刻意写成一个文件，方便对照公式读。

一次更新做五件事：

1. 用当前策略对同一道题采样 G 条回答。
2. 用规则给每条回答打分：抽出 #### 后的整数，答对 1，格式对但答错 0.1，没有格式 0。
3. 优势只在这一组里比较：A = (r - mean) / std。全对或全错时标准差为 0，这一步没有信号，跳过。
4. 损失是逐 token 的 GRPO。ratio 为 1 时 clip 不起作用，这就是 on-policy 的那一次更新::

       ratio = exp(log π - log π_old)
       L = mean( -min(ratio * A, clip(ratio) * A) + beta * KL(π || π_ref) )

   KL 用 Schulman 的 k3 估计：exp(log π_ref - log π) - (log π_ref - log π) - 1。
   对 token 取平均，这样 beta=0.04 和论文里的系数是同一个尺度。
5. 4-bit 底座冻结，只更新 LoRA。参考策略是把 LoRA scale 设为 0，也就是原来的 Instruct。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers

_ANSWER = re.compile(r"####\s*(-?\d+)")


def extract_answer(text: str) -> int | None:
    found = _ANSWER.findall(text)
    if not found:
        return None
    return int(found[-1])


def completion_reward(text: str, gold: int) -> float:
    predicted = extract_answer(text)
    if predicted is None:
        return 0.0
    if predicted == int(gold):
        return 1.0
    return 0.1


def group_advantages(rewards: list[float], eps: float = 1e-8) -> list[float] | None:
    """组内标准化。奖励完全一样时返回 None，调用方跳过这次更新。"""
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = variance**0.5
    if std < eps:
        return None
    return [(reward - mean) / (std + eps) for reward in rewards]


def _masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    return mx.sum(values * mask) / (mx.sum(mask) + 1e-8)


def grpo_loss_from_logprobs(
    logprobs: mx.array,
    old_logprobs: mx.array,
    ref_logprobs: mx.array,
    advantages: mx.array,
    mask: mx.array,
    beta: float = 0.04,
    clip_eps: float = 0.2,
) -> tuple[mx.array, dict[str, float]]:
    """逐 token GRPO。logprobs 的形状是 [组大小, token 数]，mask 为 0 的位置是 padding。"""
    logprobs = logprobs.astype(mx.float32)
    old_logprobs = old_logprobs.astype(mx.float32)
    ref_logprobs = ref_logprobs.astype(mx.float32)
    advantages = advantages.astype(mx.float32)
    mask = mask.astype(mx.float32)

    ratio = mx.exp(logprobs - old_logprobs)
    clipped = mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    advantage = advantages[:, None]
    policy = -mx.minimum(ratio * advantage, clipped * advantage)
    # k3 估计永远非负，比直接相减的 log π - log π_ref 稳。
    delta = ref_logprobs - logprobs
    kl = mx.exp(delta) - delta - 1.0
    loss = _masked_mean(policy + beta * kl, mask)
    stats = {
        "ratio_mean": float(_masked_mean(mx.stop_gradient(ratio), mask)),
        "kl_mean": float(_masked_mean(mx.stop_gradient(kl), mask)),
        "policy_loss": float(_masked_mean(mx.stop_gradient(policy), mask)),
    }
    return loss, stats


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_question(question: str) -> str:
    return (
        "请解答下面的数学题。最后一行只写最终整数答案，格式必须是 #### 整数。\n\n"
        "示例：题目：1+1 等于多少？\n"
        "1+1=2\n"
        "#### 2\n\n"
        f"题目：{question}"
    )


def prompt_tokens(tokenizer, question: str) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": format_question(question)}],
        add_generation_prompt=True,
        tokenize=True,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(token) for token in token_ids]


def generate_completion(
    model,
    prompt: list[int],
    sampler,
    max_new_tokens: int,
    eos_ids: set[int],
) -> list[int]:
    tokens: list[int] = []
    for token, _logprobs in generate_step(
        mx.array(prompt),
        model,
        max_tokens=max_new_tokens,
        sampler=sampler,
    ):
        token = int(token)
        tokens.append(token)
        if token in eos_ids:
            break
    mx.clear_cache()
    return tokens


def forward_logits(model, tokens: mx.array) -> mx.array:
    logits = model(tokens)
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


def completion_logprobs(model, sequence: list[int], prompt_len: int) -> mx.array:
    """sequence 是 prompt 接 completion。返回每个 completion token 的对数概率。"""
    tokens = mx.array(sequence)[None]
    logits = forward_logits(model, tokens)
    shifted_logits = logits[:, :-1, :]
    logprobs = shifted_logits - mx.logsumexp(shifted_logits, axis=-1, keepdims=True)
    targets = tokens[:, 1:]
    token_logprobs = mx.take_along_axis(
        logprobs, targets[:, :, None], axis=-1
    ).squeeze(-1)
    return token_logprobs[0, prompt_len - 1 :]


def pad_rows(rows: list[mx.array]) -> tuple[mx.array, mx.array]:
    width = max(int(row.shape[0]) for row in rows)
    logprobs = []
    mask = []
    for row in rows:
        length = int(row.shape[0])
        if length < width:
            row = mx.concatenate([row, mx.zeros((width - length,), dtype=row.dtype)])
        logprobs.append(row)
        mask.append(mx.array([1.0] * length + [0.0] * (width - length)))
    return mx.stack(logprobs), mx.stack(mask)


def group_logprobs(model, sequences: list[list[int]], prompt_lens: list[int]):
    rows = [
        completion_logprobs(model, sequence, prompt_len)
        for sequence, prompt_len in zip(sequences, prompt_lens)
    ]
    return pad_rows(rows)


@contextmanager
def temporary_lora_scale(model, scale: float):
    layers = [module for _, module in model.named_modules() if isinstance(module, LoRALinear)]
    previous = [layer.scale for layer in layers]
    try:
        for layer in layers:
            layer.scale = scale
        yield
    finally:
        for layer, old in zip(layers, previous):
            layer.scale = old


@dataclass
class Rollout:
    sequences: list[list[int]]
    prompt_lens: list[int]
    rewards: list[float]
    advantages: list[float]
    sample_text: str
    old_logprobs: mx.array
    ref_logprobs: mx.array


def rollout_group(
    model,
    tokenizer,
    question: str,
    gold: int,
    group_size: int,
    sampler,
    max_new_tokens: int,
    eos_ids: set[int],
    lora_scale: float,
) -> tuple[list[float], str] | Rollout:
    prompt = prompt_tokens(tokenizer, question)
    sequences = []
    texts = []
    rewards = []
    # 采样和旧策略对数概率都走当前 LoRA。参考策略再把 scale 设为 0。
    with temporary_lora_scale(model, lora_scale):
        for _ in range(group_size):
            completion = generate_completion(
                model, prompt, sampler, max_new_tokens, eos_ids
            )
            text = tokenizer.decode(completion)
            sequences.append(prompt + completion)
            texts.append(text)
            rewards.append(completion_reward(text, gold))

        advantages = group_advantages(rewards)
        preview = texts[0].replace("\n", " ")[:180]
        if advantages is None:
            return rewards, preview

        prompt_lens = [len(prompt)] * group_size
        old_logprobs, _ = group_logprobs(model, sequences, prompt_lens)
        with temporary_lora_scale(model, 0.0):
            ref_logprobs, _ = group_logprobs(model, sequences, prompt_lens)
    old_logprobs = mx.stop_gradient(old_logprobs)
    ref_logprobs = mx.stop_gradient(ref_logprobs)
    mx.eval(old_logprobs, ref_logprobs)
    return Rollout(
        sequences=sequences,
        prompt_lens=prompt_lens,
        rewards=rewards,
        advantages=advantages,
        sample_text=preview,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
    )


def grad_global_norm(grads) -> mx.array:
    total = mx.array(0.0)
    for _, grad in tree_flatten(grads):
        total = total + mx.sum(grad.astype(mx.float32) ** 2)
    return mx.sqrt(total)


def clip_grads(grads, max_norm: float):
    norm = grad_global_norm(grads)
    scale = mx.minimum(1.0, max_norm / (norm + 1e-8))
    return tree_map(lambda grad: grad * scale, grads), norm


def trainable_count(model) -> int:
    return sum(array.size for _, array in tree_flatten(model.trainable_parameters()))


def evaluate(model, tokenizer, rows, max_new_tokens: int, eos_ids: set[int]) -> float:
    sampler = make_sampler(temp=0.0)
    correct = 0
    for row in rows:
        prompt = prompt_tokens(tokenizer, row["question"])
        completion = generate_completion(
            model, prompt, sampler, max_new_tokens, eos_ids
        )
        text = tokenizer.decode(completion)
        if extract_answer(text) == int(row["answer"]):
            correct += 1
    return correct / len(rows)


def save_adapter(model, path: Path, config: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path / "adapters.safetensors"), weights)
    (path / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train(args) -> None:
    mx.random.seed(args.seed)
    random.seed(args.seed)

    print(f"加载 {args.model}")
    model, tokenizer = load(args.model)
    model.freeze()
    lora_config = {"rank": args.lora_rank, "scale": args.lora_scale, "dropout": 0.0}
    linear_to_lora_layers(model, args.num_layers, lora_config)
    model.eval()
    print(f"可训练参数: {trainable_count(model) / 1e6:.3f}M（只含 LoRA，4-bit 底座已冻结）")

    train_rows = load_jsonl(Path(args.train))
    test_rows = load_jsonl(Path(args.test))
    eos_ids = set(tokenizer.eos_token_ids)
    sampler = make_sampler(temp=args.temperature)

    print("训练前贪心评测")
    before = evaluate(model, tokenizer, test_rows, args.max_new_tokens, eos_ids)
    print(f"训练前准确率: {before:.3f} ({len(test_rows)} 题)")

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    order = list(range(len(train_rows)))
    random.shuffle(order)
    cursor = 0

    def next_rows(count: int) -> list[dict]:
        nonlocal cursor
        chosen = []
        for _ in range(count):
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            chosen.append(train_rows[order[cursor]])
            cursor += 1
        return chosen

    for step in range(1, args.steps + 1):
        mx.reset_peak_memory()
        rollouts: list[Rollout] = []
        skipped_rewards: list[float] = []
        skipped_preview = ""
        for row in next_rows(args.prompts_per_step):
            group = rollout_group(
                model,
                tokenizer,
                row["question"],
                int(row["answer"]),
                args.group_size,
                sampler,
                args.max_new_tokens,
                eos_ids,
                args.lora_scale,
            )
            if isinstance(group, Rollout):
                rollouts.append(group)
            else:
                rewards, preview = group
                skipped_rewards.extend(rewards)
                skipped_preview = preview

        if not rollouts:
            mean = sum(skipped_rewards) / len(skipped_rewards)
            print(
                f"step {step}/{args.steps} 跳过：这一步每组奖励都相同，"
                f"平均奖励 {mean:.3f}，没有相对优势"
            )
            print(f"  样例: {skipped_preview}")
            continue

        def loss_fn():
            losses = []
            stats = []
            for group in rollouts:
                current, mask = group_logprobs(model, group.sequences, group.prompt_lens)
                loss, group_stats = grpo_loss_from_logprobs(
                    current,
                    group.old_logprobs,
                    group.ref_logprobs,
                    mx.array(group.advantages),
                    mask,
                    beta=args.beta,
                    clip_eps=args.clip_eps,
                )
                losses.append(loss)
                stats.append(group_stats)
            return sum(losses) / len(losses), stats

        # value_and_grad 只对标量求导，统计量放在闭包里。
        captured: dict = {}

        def scalar_loss():
            loss, stats = loss_fn()
            captured["stats"] = stats
            return loss

        grad_fn = nn.value_and_grad(model, scalar_loss)
        last_norm = 0.0
        last_loss = 0.0
        for _ in range(args.num_iterations):
            loss, grads = grad_fn()
            grads, norm = clip_grads(grads, args.max_grad_norm)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss, norm)
            last_norm = float(norm)
            last_loss = float(loss)

        stats = captured["stats"]
        rewards = [reward for group in rollouts for reward in group.rewards]
        mean_reward = sum(rewards) / len(rewards)
        reward_var = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        ratio = sum(item["ratio_mean"] for item in stats) / len(stats)
        kl = sum(item["kl_mean"] for item in stats) / len(stats)
        peak_gb = mx.get_peak_memory() / (1024**3)
        print(
            f"step {step}/{args.steps} reward={mean_reward:.3f} "
            f"reward_std={reward_var ** 0.5:.3f} loss={last_loss:.4f} "
            f"kl={kl:.4f} ratio={ratio:.3f} grad_norm={last_norm:.4f} "
            f"peak_gb={peak_gb:.2f}"
        )
        print(f"  样例: {rollouts[0].sample_text}")

    print("训练后贪心评测")
    after = evaluate(model, tokenizer, test_rows, args.max_new_tokens, eos_ids)
    print(f"训练前准确率: {before:.3f}  训练后准确率: {after:.3f}")

    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": args.num_layers,
        "lora_parameters": lora_config,
        "model": args.model,
    }
    save_adapter(model, Path(args.adapter_path), adapter_config)
    print(f"适配器已写到 {args.adapter_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="在 Mac 上用 MLX 从 Instruct 模型做 GRPO")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--test", default="data/test.jsonl")
    parser.add_argument("--adapter-path", default="adapters")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=16.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
