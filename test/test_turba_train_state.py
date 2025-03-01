from unittest import TestCase

import numpy as np
from flax import linen as nn

from turbanet import TurbaTrainState


class TestModule(nn.Module):
    @nn.compact
    def __call__(self, x):  # noqa ANN001
        x = nn.Dense(8)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = nn.softmax(x)
        return x


class TestTurbaTrainState(TestCase):
    def setUp(self) -> None:
        self.swarm1 = TurbaTrainState.swarm(TestModule(), swarm_size=10, input_size=3)
        self.swarm2 = TurbaTrainState.swarm(TestModule(), swarm_size=10, input_size=3)

    def test_len(self) -> None:
        self.assertEqual(len(self.swarm1), 10)
        self.assertEqual(len(self.swarm2), 10)

    def test_shape(self) -> None:
        self.assertTupleEqual(self.swarm1.shape, (10, 3, 8))
        self.assertTupleEqual(self.swarm2.shape, (10, 3, 8))

    def test_add(self) -> None:
        added_swarms = self.swarm1 + self.swarm2
        self.assertEqual(len(added_swarms), 10)
        self.assertTupleEqual(added_swarms.shape, (10, 3, 8))
        for key in added_swarms.params.keys():
            # Add params together
            bias = self.swarm1.params[key]["bias"] + self.swarm2.params[key]["bias"]
            kernel = self.swarm1.params[key]["kernel"] + self.swarm2.params[key]["kernel"]

            # Test shape is correct
            self.assertTupleEqual(bias.shape, (10, 8))
            self.assertTupleEqual(kernel.shape, (10, 8))

            # Assert the params are correct
            np.testing.assert_array_equal(added_swarms.params[key]["bias"], bias)
            np.testing.assert_array_equal(added_swarms.params[key]["kernel"], kernel)

        # Assert the step and count are correct
        np.testing.assert_array_equal(self.swarm1.step, added_swarms.step)
        np.testing.assert_array_equal(
            self.swarm1.opt_state[0].count, added_swarms.opt_state[0].count
        )

    def test_append(self) -> None:
        appended_swarm = self.swarm1.append(self.swarm2)
        self.assertEqual(len(appended_swarm), 20)
        self.assertTupleEqual(appended_swarm.shape, (20, 3, 8))
        for key in appended_swarm.params.keys():
            # Add params together
            bias1 = self.swarm1.params[key]["bias"]
            bias2 = self.swarm2.params[key]["bias"]
            kernel1 = self.swarm1.params[key]["kernel"]
            kernel2 = self.swarm2.params[key]["kernel"]

            # Assert the params are correct
            np.testing.assert_array_equal(appended_swarm.params[key]["bias"][:10], bias1)
            np.testing.assert_array_equal(appended_swarm.params[key]["bias"][10:], bias2)
            np.testing.assert_array_equal(appended_swarm.params[key]["kernel"][:10], kernel1)
            np.testing.assert_array_equal(appended_swarm.params[key]["kernel"][10:], kernel2)

    def test_merge(self) -> None:
        merged_swarm = self.swarm1.merge()
        self.assertEqual(len(merged_swarm), 1)
        self.assertTupleEqual(merged_swarm.shape, (3, 8))
        for key in merged_swarm.params.keys():
            # Add params together
            bias = np.mean(self.swarm1.params[key]["bias"], axis=0)
            kernel = np.mean(self.swarm1.params[key]["kernel"], axis=0)

            # Assert the params are correct
            np.testing.assert_array_equal(merged_swarm.params[key]["bias"], bias)
            np.testing.assert_array_equal(merged_swarm.params[key]["kernel"], kernel)

        # Assert the step and count are correct
        self.assertEqual(self.swarm1.step[0], merged_swarm.step)
        self.assertEqual(self.swarm1.opt_state[0].count[0], merged_swarm.opt_state[0].count)

    def test_predict(self) -> None:
        # Multiple networks, Single batch
        input_data = np.random.rand(10, 1, 5, 3)
        self.swarm1.predict(input_data)

        # Multiple networks, Multi-batch
        input_data = np.random.rand(10, 4, 5, 3)
        self.swarm1.predict(input_data)

        # Single network, single batch
        self.single_network = TurbaTrainState.swarm(TestModule(), swarm_size=1, input_size=3)
        input_data = np.random.rand(5, 3)
        self.single_network.predict(input_data)

        # Single network, multi-batch
        input_data = np.random.rand(8, 5, 3)
        self.single_network.predict(input_data)
