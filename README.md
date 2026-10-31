# 在 Mac 上做 GRPO

这是一份能读完的 Group Relative Policy Optimization 练习。起点是已经会跟指令的 Qwen2.5 Instruct，不是 Base 模型。训练在 Apple 芯片上用 MLX 完成，只更新 LoRA，4-bit 底座冻结。

Unsloth 和 Hugging Face TRL 的 GRPO 主路径按 NVIDIA GPU 来写。这台机器上用 MLX。

## 一次更新在算什么

GRPO 没有 PPO 的价值网络。同一道题采样一组回答，优势只在这组里面比。

```
ratio = exp(log π - log π_old)
L = mean( -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) + β * KL(π || π_ref) )
A = (r - mean(r)) / std(r)
```

KL 用 Schulman 的 k3 估计，对 completion token 取平均：

```
KL = exp(log π_ref - log π) - (log π_ref - log π) - 1
```

五件对应的事：

- 奖励是规则，不是另一个模型。回答里最后一个 `####` 后面的整数等于标准答案得 1，有这个格式但数字错得 0.1，抽不出答案得 0。
- 一组里全对或全错时标准差是 0，优势没有定义，这一步跳过。
- 参考策略 π_ref 是把 LoRA scale 设为 0 的同一个模型，也就是原来的 Instruct。不另存一份权重。
- 默认每个 rollout 只优化 1 次。这时 π 和 π_old 是同一个策略，日志里的 `ratio` 是 1，clip 写在损失里但不会改变这次更新。要把 clip 真正用起来，把 `--num-iterations` 调到 2 以上。
- 损失对 token 取平均，所以 `--beta 0.04` 和论文里的 KL 系数是同一个尺度。若改成把整段对数概率加起来，长回答会主导梯度。

## 准备环境

系统自带的 Homebrew Python 不要直接装包。在仓库里建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

不下载模型也可以先核对奖励和损失：

```bash
python -m unittest test_grpo.py
```

## 跑第一次训练

默认是 `mlx-community/Qwen2.5-0.5B-Instruct-4bit`，最后 8 层 LoRA（rank 8，scale 16），每步 2 道题、每题 4 条采样，温度 0.9，最多新生成 128 个 token，学习率 `1e-5`，KL 系数 `0.04`，40 步。题目在 `data/train.jsonl`（60 道）和 `data/test.jsonl`（20 道），都是整数答案的中文算术。

```bash
python grpo_train.py
```

如果访问 `huggingface.co` 超时，先用镜像再跑同一条命令：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

0.5B 是为了让一步在十几秒量级，方便改奖励、看日志。48GB 统一内存后来可以换更大的模型，第一次不要上。

适配器写到 `adapters/`。之后可以用 mlx-lm 加载：

```bash
python -m mlx_lm generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter-path adapters \
  --prompt "请解答。最后一行写成 #### 整数。题目：3 个班级各有 28 人，又转来 4 人，现在一共多少人？"
```

## 日志怎么读

```
step 3/40 reward=0.325 reward_std=0.419 loss=-0.0123 kl=0.0041 ratio=1.000 grad_norm=0.8421 peak_gb=1.20
  样例: 3+4=7 #### 7 ...
```

- `reward` / `reward_std`：这一步所有采样的平均奖励和标准差。刚开始可以几乎不动。
- 出现「跳过」：这一步每组奖励都一样，没有相对好坏，梯度是 0。温度够、题目有的能答对有的不能，才会产生更新。
- `loss`：上面的 L。优势为正的 token 会把损失往负的方向拉，所以损失变负并不代表训练坏了。
- `kl`：当前策略相对原始 Instruct 的平均 token KL。第一步通常接近 0，因为 LoRA 的 B 矩阵初始是 0，策略还等于 Instruct。它在后面迅速变大，同时回答变空或乱码时，把学习率降到 `3e-6`，或把 `--beta` 提到 `0.1`。
- `ratio`：重要性采样比。默认是 1，说明这次是 on-policy，clip 没有生效。
- `grad_norm`：裁剪前的 LoRA 梯度范数。大于 0 说明这次真的在更新适配器。
- `peak_gb`：这一步的统一内存峰值。
- 训练前后各有一次贪心准确率，用的是 `data/test.jsonl`。40 步、0.5B 只为了确认循环可信，不是为了刷分。

## 换到 1.5B

循环确认之后只改模型名，并把生成长度收短一点。3B 仍然用 4-bit 加 LoRA，组大小保持 4。

```bash
python grpo_train.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --max-new-tokens 96
```

换奖励时改 `completion_reward`。换题目时保持 jsonl 里是 `question` 和整数 `answer`。

## 1B 全参数

`grpo_full_train.py` 用 `mlx-community/Llama-3.2-1B-Instruct-bf16`（约 2.5GB）。全部 bf16 权重都更新，参考策略是训练开始时复制的初始权重。学习率是 `1e-6`，比 LoRA 小一个数量级。结果写到 `checkpoints/llama-1b-grpo/`，这是一份完整模型，不是适配器。

```bash
export HF_ENDPOINT=https://hf-mirror.com
python grpo_full_train.py
```

加载训练结果：

```bash
python -m mlx_lm generate \
  --model checkpoints/llama-1b-grpo \
  --prompt "请解答。最后一行写成 #### 整数。题目：3 个班级各有 28 人，又转来 4 人，现在一共多少人？"
```
