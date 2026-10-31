"""Llama-3.2-1B-Instruct bf16 上的全参数 GRPO。

采样、奖励、组内优势和损失与 grpo_train.py 相同。差别是每一层权重都更新：
加载的是未量化的 bf16，不做 LoRA。参考策略是训练开始时复制出来的初始权重，
算 KL 时临时换上这份副本，算完再换回正在训练的权重。
"""

from __future__ import annotations

import argparse
import random
import shutil
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import hf_repo_to_path, save_config, save_model

from grpo_train import (
    Rollout,
    clip_grads,
    completion_reward,
    evaluate,
    generate_completion,
    grpo_loss_from_logprobs,
    group_advantages,
    group_logprobs,
    load_jsonl,
    prompt_tokens,
    trainable_count,
)

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-bf16"


def cast_floating(model, dtype) -> None:
    model.update(
        tree_map(
            lambda array: array.astype(dtype)
            if mx.issubdtype(array.dtype, mx.floating)
            else array,
            model.parameters(),
        )
    )
    mx.eval(model.parameters())


def copy_parameters(model) -> dict:
    """复制当前全部参数。之后模型被更新，这份副本保持训练开始时的 Instruct。"""
    copied = tree_map(lambda array: mx.array(array), model.parameters())
    mx.eval(copied)
    return copied


@contextmanager
def using_parameters(model, params):
    previous = model.parameters()
    model.update(params)
    try:
        yield
    finally:
        model.update(previous)


def parameter_sum(model) -> float:
    total = mx.array(0.0)
    for _, array in tree_flatten(model.parameters()):
        total = total + mx.sum(array.astype(mx.float32))
    return float(total)


def rollout_group(
    model,
    tokenizer,
    question: str,
    gold: int,
    group_size: int,
    sampler,
    max_new_tokens: int,
    eos_ids: set[int],
    ref_params: dict,
) -> tuple[list[float], str] | Rollout:
    prompt = prompt_tokens(tokenizer, question)
    sequences = []
    texts = []
    rewards = []
    for _ in range(group_size):
        completion = generate_completion(model, prompt, sampler, max_new_tokens, eos_ids)
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
    with using_parameters(model, ref_params):
        ref_logprobs, _ = group_logprobs(model, sequences, prompt_lens)
    old_logprobs = mx.stop_gradient(old_logprobs)
    ref_logprobs = mx.stop_gradient(ref_logprobs)
    mx.eval(old_logprobs, ref_logprobs)
    mx.clear_cache()
    return Rollout(
        sequences=sequences,
        prompt_lens=prompt_lens,
        rewards=rewards,
        advantages=advantages,
        sample_text=preview,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
    )


def accumulate_grads(groups: list[Rollout], model, beta: float, clip_eps: float):
    """一组一组反向传播，避免两道题的计算图同时留在内存里。"""
    total = None
    losses = []
    stats = []
    weight = 1.0 / len(groups)

    for group in groups:

        def scalar_loss(group=group):
            current, mask = group_logprobs(model, group.sequences, group.prompt_lens)
            loss, group_stats = grpo_loss_from_logprobs(
                current,
                group.old_logprobs,
                group.ref_logprobs,
                mx.array(group.advantages),
                mask,
                beta=beta,
                clip_eps=clip_eps,
            )
            stats.append(group_stats)
            return loss

        loss, grads = nn.value_and_grad(model, scalar_loss)()
        mx.eval(loss, grads)
        losses.append(loss)
        scaled = tree_map(lambda grad: grad * weight, grads)
        total = scaled if total is None else tree_map(lambda left, right: left + right, total, scaled)
        mx.eval(total)
        mx.clear_cache()

    return sum(losses) / len(losses), total, stats


def save_full_model(model, tokenizer, config: dict, src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    save_model(dst, model, donate_model=False)
    save_config(config, dst / "config.json")
    tokenizer.save_pretrained(dst)
    for name in ("generation_config.json", "special_tokens_map.json"):
        source = src / name
        if source.exists():
            shutil.copy(source, dst / name)


def train(args) -> None:
    mx.random.seed(args.seed)
    random.seed(args.seed)

    print(f"加载 {args.model}")
    model, tokenizer, config = load(args.model, return_config=True)
    model.unfreeze()
    cast_floating(model, mx.bfloat16)
    model.eval()
    count = trainable_count(model)
    sample_dtype = next(array.dtype for _, array in tree_flatten(model.parameters()))
    print(f"可训练参数: {count / 1e9:.3f}B（{sample_dtype} 全参数，没有冻结层）")
    ref_params = copy_parameters(model)
    before_sum = parameter_sum(model)
    print(f"已复制初始权重作为参考策略，参数和 {before_sum:.3f}")

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
                ref_params,
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

        last_loss = None
        last_norm = 0.0
        stats = []
        for _ in range(args.num_iterations):
            loss, grads, stats = accumulate_grads(
                rollouts, model, args.beta, args.clip_eps
            )
            grads, norm = clip_grads(grads, args.max_grad_norm)
            grads = tree_map(lambda grad: grad.astype(mx.bfloat16), grads)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss, norm)
            last_loss = loss
            last_norm = float(norm)

        rewards = [reward for group in rollouts for reward in group.rewards]
        mean_reward = sum(rewards) / len(rewards)
        reward_var = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        ratio = sum(item["ratio_mean"] for item in stats) / len(stats)
        kl = sum(item["kl_mean"] for item in stats) / len(stats)
        peak_gb = mx.get_peak_memory() / (1024**3)
        print(
            f"step {step}/{args.steps} reward={mean_reward:.3f} "
            f"reward_std={reward_var ** 0.5:.3f} loss={float(last_loss):.4f} "
            f"kl={kl:.4f} ratio={ratio:.3f} grad_norm={last_norm:.4f} "
            f"peak_gb={peak_gb:.2f}"
        )
        print(f"  样例: {rollouts[0].sample_text}")

    print("训练后贪心评测")
    after = evaluate(model, tokenizer, test_rows, args.max_new_tokens, eos_ids)
    after_sum = parameter_sum(model)
    print(f"训练前准确率: {before:.3f}  训练后准确率: {after:.3f}")
    print(f"参数和 {before_sum:.3f} -> {after_sum:.3f}")

    src = hf_repo_to_path(args.model)
    save_full_model(model, tokenizer, config, src, Path(args.save_path))
    print(f"全量权重已写到 {args.save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Llama-3.2-1B-Instruct bf16 全参数 GRPO")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--test", default="data/test.jsonl")
    parser.add_argument("--save-path", default="checkpoints/llama-1b-grpo")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
