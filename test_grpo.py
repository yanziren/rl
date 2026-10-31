"""GRPO 里不依赖模型的部分：抽答案、奖励、组内优势、带 clip 的损失。"""

import math
import unittest

import mlx.core as mx

from grpo_train import (
    completion_reward,
    extract_answer,
    group_advantages,
    grpo_loss_from_logprobs,
)


class ExtractAndRewardTest(unittest.TestCase):
    def test_extracts_integer_after_marker(self):
        self.assertEqual(extract_answer("先算加法。\n#### 42\n"), 42)

    def test_extracts_negative(self):
        self.assertEqual(extract_answer("#### -3"), -3)

    def test_uses_the_last_marker(self):
        self.assertEqual(extract_answer("#### 7\n检查后 #### 8"), 8)

    def test_missing_marker_is_none(self):
        self.assertIsNone(extract_answer("答案是 7"))

    def test_reward_correct_wrong_and_missing(self):
        self.assertEqual(completion_reward("#### 19", 19), 1.0)
        self.assertEqual(completion_reward("#### 20", 19), 0.1)
        self.assertEqual(completion_reward("我不知道", 19), 0.0)


class AdvantageTest(unittest.TestCase):
    def test_higher_reward_gets_positive_advantage(self):
        advantages = group_advantages([0.0, 0.1, 1.0, 0.1])
        self.assertIsNotNone(advantages)
        self.assertAlmostEqual(sum(advantages), 0.0, places=5)
        self.assertGreater(advantages[2], 0.0)
        self.assertLess(advantages[0], 0.0)

    def test_identical_rewards_have_no_signal(self):
        self.assertIsNone(group_advantages([0.1, 0.1, 0.1, 0.1]))


class LossTest(unittest.TestCase):
    def test_on_policy_ratio_is_one_and_loss_is_negative_advantage(self):
        logprobs = mx.zeros((1, 2))
        advantages = mx.array([1.0])
        mask = mx.ones((1, 2))
        loss, stats = grpo_loss_from_logprobs(
            logprobs, logprobs, logprobs, advantages, mask, beta=0.04, clip_eps=0.2
        )
        self.assertAlmostEqual(float(loss), -1.0, places=5)
        self.assertAlmostEqual(stats["ratio_mean"], 1.0, places=5)
        self.assertAlmostEqual(stats["kl_mean"], 0.0, places=5)

    def test_kl_uses_schulman_k3(self):
        logprobs = mx.zeros((1, 1))
        ref = mx.array([[-1.0]])
        advantages = mx.zeros((1,))
        mask = mx.ones((1, 1))
        loss, stats = grpo_loss_from_logprobs(
            logprobs, logprobs, ref, advantages, mask, beta=1.0, clip_eps=0.2
        )
        expected = math.exp(-1.0) - (-1.0) - 1.0
        self.assertAlmostEqual(float(loss), expected, places=5)
        self.assertAlmostEqual(stats["kl_mean"], expected, places=5)

    def test_positive_advantage_clips_large_ratio(self):
        logprobs = mx.array([[math.log(2.0)]])
        old = mx.zeros((1, 1))
        advantages = mx.array([1.0])
        mask = mx.ones((1, 1))
        loss, stats = grpo_loss_from_logprobs(
            logprobs, old, logprobs, advantages, mask, beta=0.0, clip_eps=0.2
        )
        self.assertAlmostEqual(float(loss), -1.2, places=5)
        self.assertAlmostEqual(stats["ratio_mean"], 2.0, places=5)

    def test_padding_is_ignored(self):
        logprobs = mx.array([[0.0, 5.0]])
        old = mx.array([[0.0, 0.0]])
        ref = logprobs
        advantages = mx.array([1.0])
        mask = mx.array([[1.0, 0.0]])
        loss, stats = grpo_loss_from_logprobs(
            logprobs, old, ref, advantages, mask, beta=0.0, clip_eps=0.2
        )
        self.assertAlmostEqual(float(loss), -1.0, places=5)
        self.assertAlmostEqual(stats["ratio_mean"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
