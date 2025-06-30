# Sinir ağının temel yapısını, katmanlarını, ağırlıklarını,
# biaslarını yöneten ve ileri/geri yayılım algoritmalarını içeren
# NeuralNetwork sınıfı.

import math
import random
from utils import (
    ACTIVATION_FUNCTIONS, LOSS_FUNCTIONS,
    multiply_row_vector_matrix, add_vectors, subtract_vectors,
    elementwise_multiply_vectors, transpose_matrix,
    multiply_scalar_vector, multiply_scalar_matrix,
    add_matrices, subtract_matrices
)

class NeuralNetwork:
    def __init__(self, loss_function_name="mean_squared_error"):
        self.layer_configs = [] 
        self.weights = [] 
        self.biases = []  
        self.neuron_outputs_z = [] 
        self.neuron_outputs_a = [] 
        self.current_input_for_forward = []
        self.loss_function_name = loss_function_name
        self.loss_func, self.loss_derivative_func = LOSS_FUNCTIONS[loss_function_name]
        self.velocity_W, self.velocity_b = [], []
        self.m_W, self.v_W, self.m_b, self.v_b = [], [], [], []
        self.adam_t = 0 

    def set_loss_function(self, loss_name):
        if loss_name in LOSS_FUNCTIONS:
            self.loss_function_name = loss_name
            self.loss_func, self.loss_derivative_func = LOSS_FUNCTIONS[loss_name]
        else: raise ValueError(f"Bilinmeyen kayıp fonksiyonu: {loss_name}")

    def _initialize_optimizer_params(self, prev_layer_neuron_count, num_neurons):
        self.velocity_W.append([[0.0 for _ in range(num_neurons)] for _ in range(prev_layer_neuron_count)])
        self.velocity_b.append([0.0 for _ in range(num_neurons)])
        self.m_W.append([[0.0 for _ in range(num_neurons)] for _ in range(prev_layer_neuron_count)])
        self.v_W.append([[0.0 for _ in range(num_neurons)] for _ in range(prev_layer_neuron_count)])
        self.m_b.append([0.0 for _ in range(num_neurons)])
        self.v_b.append([0.0 for _ in range(num_neurons)])

    def configure_network(self, input_size, layer_configs_from_gui, custom_weights=None, custom_biases=None):
        self.layer_configs = layer_configs_from_gui
        self.weights, self.biases = [], []
        self.velocity_W, self.velocity_b = [], []
        self.m_W, self.v_W, self.m_b, self.v_b = [], [], [], []
        self.adam_t = 0
        prev_layer_neuron_count = input_size
        for i, (num_neurons, _) in enumerate(self.layer_configs):
            if custom_weights and i < len(custom_weights):
                layer_weights = [[float(w_val) for w_val in w_row] for w_row in custom_weights[i]] 
                if len(layer_weights) != prev_layer_neuron_count or (layer_weights and len(layer_weights[0]) != num_neurons):
                    raise ValueError(f"Katman {i+1} özel W boyutları ({len(layer_weights)}x{len(layer_weights[0]) if layer_weights else 0}) != beklenen ({prev_layer_neuron_count}x{num_neurons}).")
            else:
                limit = math.sqrt(6 / (prev_layer_neuron_count + num_neurons)) if (prev_layer_neuron_count + num_neurons > 0) else 0.5
                layer_weights = [[random.uniform(-limit, limit) for _ in range(num_neurons)] for _ in range(prev_layer_neuron_count)]
            if custom_biases and i < len(custom_biases):
                layer_biases = [float(b_val) for b_val in custom_biases[i]] 
                if len(layer_biases) != num_neurons:
                     raise ValueError(f"Katman {i+1} özel B boyutu ({len(layer_biases)}) != beklenen ({num_neurons}).")
            else: layer_biases = [random.uniform(-0.1, 0.1) for _ in range(num_neurons)]
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)
            self._initialize_optimizer_params(prev_layer_neuron_count, num_neurons)
            prev_layer_neuron_count = num_neurons

    def get_activation_func_obj(self, layer_idx): 
        _, activation_str = self.layer_configs[layer_idx]
        return ACTIVATION_FUNCTIONS[activation_str][0]

    def get_activation_derivative_func_obj(self, layer_idx): 
        _, activation_str = self.layer_configs[layer_idx]
        if ACTIVATION_FUNCTIONS[activation_str][1] is None and activation_str == "softmax":
            return lambda x: [1.0] * len(x) if isinstance(x, list) else 1.0 
        return ACTIVATION_FUNCTIONS[activation_str][1]

    def forward_pass_generator(self, inputs, detailed_steps=False):
        self.current_input_for_forward = list(inputs)
        self.neuron_outputs_z, self.neuron_outputs_a = [], [list(inputs)] 
        current_activations = list(inputs)
        yield {"type": "input_layer", "layer_index": -1, "outputs": list(current_activations), "num_neurons": len(current_activations)}
        for i in range(len(self.weights)): 
            layer_weights, layer_biases = self.weights[i], self.biases[i]   
            num_current_neurons, num_prev_neurons = len(layer_biases), len(current_activations)
            activation_name = self.layer_configs[i][1]
            z_values = [0.0] * num_current_neurons
            if detailed_steps:
                for j in range(num_current_neurons): 
                    neuron_z_unbiased = 0.0
                    for k in range(num_prev_neurons): 
                        weight, activation_prev = layer_weights[k][j], current_activations[k]
                        product = activation_prev * weight; neuron_z_unbiased += product
                        yield {"type": "weight_multiplication", "layer_index": i, "neuron_index": j, "prev_neuron_index": k, "weight": weight, "prev_activation": activation_prev, "product": product, "current_sum_for_neuron_z": neuron_z_unbiased}
                    z_values[j] = neuron_z_unbiased + layer_biases[j]
                    yield {"type": "bias_addition", "layer_index": i, "neuron_index": j, "z_unbiased": neuron_z_unbiased, "bias": layer_biases[j], "z_final": z_values[j]}
            else: z_values_unbiased = multiply_row_vector_matrix(current_activations, layer_weights); z_values = add_vectors(z_values_unbiased, layer_biases)
            activation_func_obj = self.get_activation_func_obj(i)
            a_values = activation_func_obj(z_values) if activation_name == "softmax" else [activation_func_obj(z) for z in z_values]
            self.neuron_outputs_z.append(list(z_values)); self.neuron_outputs_a.append(list(a_values))
            yield {"type": "layer_activation", "layer_index": i, "inputs_to_layer": list(current_activations), "z_values": list(z_values), "a_values": list(a_values), "activation_function": activation_name, "num_neurons": num_current_neurons}
            current_activations = list(a_values)
        yield {"type": "forward_pass_complete", "final_output": list(current_activations)}

    def backward_pass_generator(self, targets, learning_rate, optimizer_params=None):
        if not self.neuron_outputs_a or len(self.neuron_outputs_a) <= 1: yield {"type": "error", "message": "İleri yayılım çalıştırılmadı."}; return
        optimizer_params = optimizer_params or {}; optimizer_type = optimizer_params.get("type", "sgd")
        beta_momentum, beta1_adam, beta2_adam, epsilon_adam = optimizer_params.get("beta", 0.9), optimizer_params.get("beta1", 0.9), optimizer_params.get("beta2", 0.999), optimizer_params.get("epsilon", 1e-8)
        if optimizer_type == "adam": self.adam_t += 1
        output_layer_idx, a_L, z_L, delta_L = len(self.weights) - 1, self.neuron_outputs_a[-1], self.neuron_outputs_z[-1], []
        if self.loss_function_name == "cross_entropy" and self.layer_configs[output_layer_idx][1] == "softmax":
            delta_L = self.loss_derivative_func(targets, a_L) 
            yield {"type": "output_delta_calculation", "layer_index": output_layer_idx, "method": "cross_entropy_with_softmax (dL/dz_L)", "a_L": list(a_L), "targets": list(targets), "delta_L": list(delta_L), "num_neurons": len(delta_L)}
        else: 
            dL_daL = subtract_vectors(a_L, targets) if self.loss_function_name == "mean_squared_error" else self.loss_derivative_func(targets, a_L)
            activation_derivative_func_obj = self.get_activation_derivative_func_obj(output_layer_idx)
            f_prime_z_L = [activation_derivative_func_obj(z) for z in z_L]
            delta_L = elementwise_multiply_vectors(dL_daL, f_prime_z_L) 
            yield {"type": "output_delta_calculation", "layer_index": output_layer_idx, "method": "elementwise_error_times_derivative (dL/dz_L)", "dL_daL": list(dL_daL), "f_prime_z_L": list(f_prime_z_L), "delta_L": list(delta_L), "num_neurons": len(delta_L)}
        deltas = [delta_L] 
        for l in range(len(self.weights) - 2, -1, -1): 
            delta_next_layer, weights_next_layer = deltas[0], self.weights[l+1] 
            error_propagated = multiply_row_vector_matrix(delta_next_layer, transpose_matrix(weights_next_layer))
            z_l, activation_derivative_func_l_obj = self.neuron_outputs_z[l], self.get_activation_derivative_func_obj(l)
            f_prime_z_l = [activation_derivative_func_l_obj(z_val) for z_val in z_l]
            delta_l = elementwise_multiply_vectors(error_propagated, f_prime_z_l)
            deltas.insert(0, delta_l) 
            yield {"type": "hidden_delta_calculation", "layer_index": l, "delta_next_layer": list(delta_next_layer), "error_propagated": list(error_propagated), "f_prime_z_l": list(f_prime_z_l), "delta_l": list(delta_l), "num_neurons": len(delta_l)}
        for l in range(len(self.weights)):
            a_prev_layer, delta_curr_layer = self.neuron_outputs_a[l], deltas[l] 
            grad_W_l, grad_b_l = [[a_prev * d_curr for d_curr in delta_curr_layer] for a_prev in a_prev_layer], delta_curr_layer 
            yield {"type": "gradient_calculation", "layer_index": l, "grad_W_l_dims": (len(grad_W_l), len(grad_W_l[0]) if grad_W_l else 0), "grad_b_l_dims": len(grad_b_l)}
            if optimizer_type == "sgd":
                self.weights[l] = subtract_matrices(self.weights[l], multiply_scalar_matrix(learning_rate, grad_W_l))
                self.biases[l] = subtract_vectors(self.biases[l], multiply_scalar_vector(learning_rate, grad_b_l))
            elif optimizer_type == "momentum":
                self.velocity_W[l] = add_matrices(multiply_scalar_matrix(beta_momentum, self.velocity_W[l]), multiply_scalar_matrix(learning_rate, grad_W_l))
                self.velocity_b[l] = add_vectors(multiply_scalar_vector(beta_momentum, self.velocity_b[l]), multiply_scalar_vector(learning_rate, grad_b_l))
                self.weights[l] = subtract_matrices(self.weights[l], self.velocity_W[l])
                self.biases[l] = subtract_vectors(self.biases[l], self.velocity_b[l])
            elif optimizer_type == "adam":
                self.m_W[l] = add_matrices(multiply_scalar_matrix(beta1_adam, self.m_W[l]), multiply_scalar_matrix(1 - beta1_adam, grad_W_l))
                self.m_b[l] = add_vectors(multiply_scalar_vector(beta1_adam, self.m_b[l]), multiply_scalar_vector(1 - beta1_adam, grad_b_l))
                grad_W_l_sq, grad_b_l_sq = [[g**2 for g in row] for row in grad_W_l], [g**2 for g in grad_b_l]
                self.v_W[l] = add_matrices(multiply_scalar_matrix(beta2_adam, self.v_W[l]), multiply_scalar_matrix(1 - beta2_adam, grad_W_l_sq))
                self.v_b[l] = add_vectors(multiply_scalar_vector(beta2_adam, self.v_b[l]), multiply_scalar_vector(1 - beta2_adam, grad_b_l_sq))
                
                # 0'a bölme hatasını önlemek için küçük bir kontrol
                denom_beta1 = (1 - beta1_adam**self.adam_t)
                denom_beta2 = (1 - beta2_adam**self.adam_t)
                if denom_beta1 == 0: denom_beta1 = 1e-8 
                if denom_beta2 == 0: denom_beta2 = 1e-8

                m_W_hat = multiply_scalar_matrix(1 / denom_beta1, self.m_W[l])
                m_b_hat = multiply_scalar_vector(1 / denom_beta1, self.m_b[l])
                v_W_hat = multiply_scalar_matrix(1 / denom_beta2, self.v_W[l])
                v_b_hat = multiply_scalar_vector(1 / denom_beta2, self.v_b[l])
                
                update_term_W = [[ (learning_rate * m_w_h) / (math.sqrt(v_w_h) + epsilon_adam) for m_w_h, v_w_h in zip(m_row, v_row)] for m_row, v_row in zip(m_W_hat, v_W_hat)]
                update_term_b = [ (learning_rate * m_b_h) / (math.sqrt(v_b_h) + epsilon_adam) for m_b_h, v_b_h in zip(m_b_hat, v_b_hat)]
                self.weights[l] = subtract_matrices(self.weights[l], update_term_W)
                self.biases[l] = subtract_vectors(self.biases[l], update_term_b)
            yield {"type": "weight_update", "layer_index": l, "optimizer_used": optimizer_type}
        yield {"type": "backward_pass_complete"}