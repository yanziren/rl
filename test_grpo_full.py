"""全参数 GRPO 里，参考策略必须是初始权重的副本，不能跟着更新走。"""

import unittest

import mlx.core as mx
import mlx.nn as nn

from grpo_full_train import cast_floating, copy_parameters, using_parameters


class SnapshotTest(unittest.TestCase):
    def test_copy_stays_at_the_initial_weights(self):
        layer = nn.Linear(2, 2, bias=False)
        original = mx.array(layer.weight)
        snapshot = copy_parameters(layer)
        layer.update({"weight": layer.weight + 1})
        mx.eval(layer.parameters())
        self.assertTrue(mx.allclose(snapshot["weight"], original))
        self.assertFalse(mx.allclose(layer.weight, original))

    def test_using_parameters_restores_the_live_weights(self):
        layer = nn.Linear(2, 2, bias=False)
        snapshot = copy_parameters(layer)
        layer.update({"weight": layer.weight + 3})
        mx.eval(layer.parameters())
        shifted = mx.array(layer.weight)
        with using_parameters(layer, snapshot):
            self.assertTrue(mx.allclose(layer.weight, snapshot["weight"]))
        self.assertTrue(mx.allclose(layer.weight, shifted))

    def test_cast_floating_changes_dtype(self):
        layer = nn.Linear(2, 2, bias=False)
        cast_floating(layer, mx.bfloat16)
        self.assertEqual(layer.weight.dtype, mx.bfloat16)


if __name__ == "__main__":
    unittest.main()
