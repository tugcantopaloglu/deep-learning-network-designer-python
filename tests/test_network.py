import copy
import unittest

from source_code.neural_network import NeuralNetwork


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.network = NeuralNetwork()
        self.network.configure_network(
            2, [(2, "linear")],
            custom_weights=[[[0.2, -0.3], [0.4, 0.5]]],
            custom_biases=[[0.1, -0.2]],
        )

    def test_detailed_and_layer_forward_have_same_result(self):
        compact = list(self.network.forward_pass_generator([1.0, 2.0]))[-1]
        detailed = list(self.network.forward_pass_generator([1.0, 2.0], True))[-1]
        self.assertEqual(compact, detailed)
        self.assertAlmostEqual(compact["final_output"][0], 1.1)
        self.assertAlmostEqual(compact["final_output"][1], 0.5)

    def test_reconfigure_clears_forward_and_optimizer_state(self):
        list(self.network.forward_pass_generator([1.0, 2.0]))
        list(self.network.backward_pass_generator([0.0, 1.0], 0.01, {"type": "adam"}))
        self.network.configure_network(1, [(1, "sigmoid")])
        self.assertEqual(self.network.neuron_outputs_z, [])
        self.assertEqual(self.network.neuron_outputs_a, [])
        self.assertEqual(self.network.current_input_for_forward, [])
        self.assertEqual(self.network.adam_t, 0)
        self.assertEqual(self.network.m_W, [[[0.0]]])
        self.assertEqual(list(self.network.backward_pass_generator([0.0], 0.01))[0]["type"], "error")

    def test_invalid_configuration_preserves_existing_network(self):
        list(self.network.forward_pass_generator([1.0, 2.0]))
        before = copy.deepcopy(self.network.__dict__)
        invalid = [
            (0, [(2, "linear")], None, None),
            (2, [], None, None),
            (2, [(0, "linear")], None, None),
            (2, [(2, "unknown")], None, None),
            (2, [(2, "linear")], [[[1, 2], [3]]], None),
            (2, [(2, "linear")], [[[1, 2], [3, 4, 5]]], None),
            (2, [(2, "linear")], [], None),
            (2, [(2, "linear")], None, [[1]]),
            (2, [(2, "linear")], [[[1, 2], [3, float("nan")]]], None),
            (2, [(2, "linear")], None, [[1, float("inf")]]),
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.network.configure_network(*args)
            self.assertEqual(self.network.__dict__, before)

    def test_configuration_does_not_alias_callers_layer_list(self):
        layers = [[1, "linear"]]
        self.network.configure_network(1, layers)
        layers[0][0] = 0
        self.assertEqual(self.network.layer_configs, [(1, "linear")])

    def test_multi_output_mse_update_matches_finite_difference(self):
        inputs, targets = [0.6, -0.4], [0.1, -0.2]
        epsilon, learning_rate = 1e-6, 0.01
        gradients = []
        for row in self.network.weights[0]:
            row_gradients = []
            for column in range(len(row)):
                original = row[column]
                row[column] = original + epsilon
                list(self.network.forward_pass_generator(inputs))
                plus = self.network.loss_func(targets, self.network.neuron_outputs_a[-1])
                row[column] = original - epsilon
                list(self.network.forward_pass_generator(inputs))
                minus = self.network.loss_func(targets, self.network.neuron_outputs_a[-1])
                row[column] = original
                row_gradients.append((plus - minus) / (2 * epsilon))
            gradients.append(row_gradients)
        original_weights = copy.deepcopy(self.network.weights[0])
        list(self.network.forward_pass_generator(inputs))
        list(self.network.backward_pass_generator(targets, learning_rate))
        for i, row in enumerate(self.network.weights[0]):
            for j, weight in enumerate(row):
                update_gradient = (original_weights[i][j] - weight) / learning_rate
                self.assertAlmostEqual(update_gradient, gradients[i][j], places=8)

    def test_mismatched_target_rejected_before_updates(self):
        list(self.network.forward_pass_generator([1.0, 2.0]))
        before = copy.deepcopy(self.network.weights)
        with self.assertRaises(ValueError):
            list(self.network.backward_pass_generator([1.0], 0.1))
        self.assertEqual(self.network.weights, before)

    def test_bad_input_rejected_in_both_forward_modes(self):
        for detailed in (False, True):
            with self.subTest(detailed=detailed), self.assertRaises(ValueError):
                list(self.network.forward_pass_generator([1.0], detailed))

    def test_incomplete_forward_does_not_update_optimizer(self):
        self.network.configure_network(2, [(2, "relu"), (1, "sigmoid")])
        forward = self.network.forward_pass_generator([1.0, 2.0])
        next(forward)
        next(forward)
        self.assertEqual(list(self.network.backward_pass_generator([1.0], 0.1, {"type": "adam"}))[0]["type"], "error")
        self.assertEqual(self.network.adam_t, 0)

    def test_cross_entropy_requires_softmax_before_optimizer_update(self):
        self.network.set_loss_function("cross_entropy")
        list(self.network.forward_pass_generator([1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "softmax"):
            list(self.network.backward_pass_generator([0.0, 1.0], 0.1, {"type": "adam"}))
        self.assertEqual(self.network.adam_t, 0)

    def test_softmax_gradients_match_finite_difference(self):
        for loss, layers in (
            ("mean_squared_error", [(2, "softmax")]),
            ("mean_squared_error", [(2, "softmax"), (2, "linear")]),
            ("cross_entropy", [(2, "softmax"), (2, "softmax")]),
        ):
            with self.subTest(loss=loss, layers=layers):
                network = NeuralNetwork(loss)
                network.configure_network(
                    2, layers,
                    custom_weights=[[[0.2, -0.3], [0.4, 0.5]]] * len(layers),
                    custom_biases=[[0.1, -0.2]] * len(layers),
                )
                inputs, targets = [0.6, -0.4], [0.0, 1.0]
                epsilon, learning_rate = 1e-6, 0.01
                numerical = []
                for matrix in network.weights:
                    gradients = []
                    for row in matrix:
                        row_gradients = []
                        for column, original in enumerate(row):
                            row[column] = original + epsilon
                            plus = list(network.forward_pass_generator(inputs))[-1]["final_output"]
                            row[column] = original - epsilon
                            minus = list(network.forward_pass_generator(inputs))[-1]["final_output"]
                            row[column] = original
                            row_gradients.append((network.loss_func(targets, plus) - network.loss_func(targets, minus)) / (2 * epsilon))
                        gradients.append(row_gradients)
                    numerical.append(gradients)
                before = copy.deepcopy(network.weights)
                list(network.forward_pass_generator(inputs))
                list(network.backward_pass_generator(targets, learning_rate))
                for layer, matrix in enumerate(network.weights):
                    for i, row in enumerate(matrix):
                        for j, weight in enumerate(row):
                            self.assertAlmostEqual((before[layer][i][j] - weight) / learning_rate, numerical[layer][i][j], places=8)


if __name__ == "__main__":
    unittest.main()
