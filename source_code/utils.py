# Sinir ağı simülatörü için temel matematiksel işlemleri,
# aktivasyon fonksiyonlarını, kayıp fonksiyonlarını ve diğer genel yardımcı
# araçları içerir.

import math

# Aktivasyon Fonksiyonları ve Türevleri
def sigmoid(x):
    try:
        if x < -700: return 0.0
        if x > 700: return 1.0
        return 1 / (1 + math.exp(-x))
    except OverflowError: 
        return 0.0 if x < 0 else 1.0

def sigmoid_derivative(x):
    s_x = sigmoid(x)
    return s_x * (1 - s_x)

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def tanh_activation(x):
    try:
        if x < -700: return -1.0
        if x > 700: return 1.0
        return math.tanh(x)
    except OverflowError:
        return -1.0 if x < 0 else 1.0

def tanh_derivative(x):
    return 1.0 - tanh_activation(x)**2 

def linear(x):
    return x

def linear_derivative(x):
    return 1.0

def softmax(x_vector):
    if not x_vector: return []
    try:
        max_x = max(x_vector)
        exp_x = [math.exp(val - max_x) for val in x_vector]
        sum_exp_x = sum(exp_x)
        if sum_exp_x == 0: 
            return [1.0 / len(x_vector) for _ in x_vector] 
        return [val / sum_exp_x for val in exp_x]
    except OverflowError:
        max_val_idx = x_vector.index(max(x_vector))
        return [1.0 if i == max_val_idx else 0.0 for i in range(len(x_vector))]

ACTIVATION_FUNCTIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh_activation, tanh_derivative),
    "linear": (linear, linear_derivative),
    "softmax": (softmax, None) 
}

# --- Kayıp Fonksiyonları ve Türevleri ---
def mean_squared_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError(f"MSE: Gerçek ({len(y_true)}) ve tahmin ({len(y_pred)}) boyutları eşleşmeli.")
    return 0.5 * sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

def mean_squared_error_derivative_for_dL_daL(y_true, y_pred):
    return [(yp - yt) / len(y_true) for yt, yp in zip(y_true, y_pred)]

def cross_entropy_loss(y_true_one_hot, y_pred_probs):
    if len(y_true_one_hot) != len(y_pred_probs):
        raise ValueError(f"CrossEntropy: Gerçek ({len(y_true_one_hot)}) ve tahmin ({len(y_pred_probs)}) boyutları eşleşmeli.")
    epsilon = 1e-12 
    loss = 0
    for i in range(len(y_true_one_hot)):
        loss -= y_true_one_hot[i] * math.log(max(y_pred_probs[i], epsilon))
    return loss 

def cross_entropy_loss_derivative_with_softmax_for_dL_dzL(y_true_one_hot, y_pred_softmax_output):
    if len(y_true_one_hot) != len(y_pred_softmax_output):
        raise ValueError(f"CE Türevi: Gerçek ({len(y_true_one_hot)}) ve tahmin ({len(y_pred_softmax_output)}) boyutları eşleşmeli.")
    return subtract_vectors(y_pred_softmax_output, y_true_one_hot)

LOSS_FUNCTIONS = {
    "mean_squared_error": (mean_squared_error, mean_squared_error_derivative_for_dL_daL),
    "cross_entropy": (cross_entropy_loss, cross_entropy_loss_derivative_with_softmax_for_dL_dzL)
}

# Vektör/Matris Yardımcı Fonksiyonları
def multiply_row_vector_matrix(row_vector, matrix):
    if not matrix: return [] 
    if not matrix[0] and not row_vector: return [] 
    if not matrix[0] and row_vector: raise ValueError("Matris sütunsuz ama vektör elemanlı.")
    if len(row_vector) != len(matrix): raise ValueError(f"Vektör boyutu ({len(row_vector)}) matrisin satır sayısıyla ({len(matrix)}) eşleşmeli.")
    num_curr_neurons = len(matrix[0])
    result = [0.0] * num_curr_neurons
    for j in range(num_curr_neurons):
        s = 0.0
        for i in range(len(row_vector)): s += row_vector[i] * matrix[i][j]
        result[j] = s
    return result

def add_vectors(v1, v2):
    if len(v1) != len(v2): raise ValueError(f"Vektör boyutları toplama için eşleşmeli ({len(v1)} vs {len(v2)}).")
    return [x + y for x, y in zip(v1, v2)]

def subtract_vectors(v1, v2):
    if len(v1) != len(v2): raise ValueError(f"Vektör boyutları çıkarma için eşleşmeli ({len(v1)} vs {len(v2)}).")
    return [x - y for x, y in zip(v1, v2)]

def elementwise_multiply_vectors(v1, v2):
    if len(v1) != len(v2): raise ValueError(f"Vektör boyutları eleman bazında çarpma için eşleşmeli ({len(v1)} vs {len(v2)}).")
    return [x * y for x, y in zip(v1, v2)]

def transpose_matrix(matrix):
    if not matrix: return []
    if not matrix[0]: return [[] for _ in matrix] 
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def multiply_scalar_vector(scalar, vector):
    return [scalar * x for x in vector]

def multiply_scalar_matrix(scalar, matrix):
    return [[scalar * val for val in row] for row in matrix]

def add_matrices(m1, m2):
    if len(m1) != len(m2) or (m1 and len(m1[0]) != len(m2[0])): raise ValueError("Matris boyutları toplama için eşleşmeli.")
    return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]

def subtract_matrices(m1, m2):
    if len(m1) != len(m2) or (m1 and len(m1[0]) != len(m2[0])): raise ValueError("Matris boyutları çıkarma için eşleşmeli.")
    return [[m1[i][j] - m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]

# Tooltip Sınıfı
import tkinter as tk
from tkinter import ttk

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(self.tooltip_window, text=self.text, background="#ffffe0", relief="solid", borderwidth=1, padding=3, font=("Calibri", 9))
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None