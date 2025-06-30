# Deep Network Design Simulator

This application is a simulator built to **understand the inner mechanics of deep‑learning neural networks**, design various network architectures, train them, and visualise their results.  
All mathematical operations are implemented **manually in Python**, helping users learn by doing.  
You can run the project either via **`main.py`** or through the provided **`.exe`** file.

---

## Features

- **Dynamic Network Configuration**  
  Define the number of neurons in the **input**, **hidden**, and **output** layers, as well as the number of hidden layers.
- **Activation Function Selection**  
  Choose a different activation function (Sigmoid, ReLU, Tanh, Linear, Softmax) for every layer except the input.
- **Loss Function Selection**  
  Switch between **Mean Squared Error (MSE)** and **Cross‑Entropy** loss.
- **Optimisation Algorithms**  
  Select among **SGD**, **Momentum**, and **Adam** optimisers.
- **Manual & Automatic Training**  
  Train the network **step‑by‑step** or **fully automatically**.
- **Step‑by‑Step Monitoring**  
  - Observe each forward/backward step in detail (weight products, bias additions, activations, deltas, gradients, weight updates).  
  - Toggle visualisation of these steps during automatic training.
- **Visualisation**  
  - Live view of the neural‑network graph.  
  - Show/hide neuron values: activation (a), weighted sum (z), bias (b).  
  - Show/hide connection weights (w).  
  - Connection thickness & colour adapt dynamically to weight magnitude.
- **Graphs**  
  - **Loss per epoch**.  
  - **Accuracy per epoch** (when using Cross‑Entropy + Softmax).  
  - **Text‑based Confusion Matrix** (when using Cross‑Entropy + Softmax).
- **Data Handling**  
  - Enter data manually.  
  - Load data from CSV.
- **Save / Load Network & Training State**  
  - Save the designed network (structure, weights, biases, optimizer state, training history) as **`.json`** and reload later.  
  - Save the network graph as **`.eps`**.
- **User Interface**  
  - Modern UI (Sun‑Valley theme support).  
  - Switch between **light** and **dark** mode.  
  - Scrollable control panel.  
  - Click on neurons to get detailed info.

---

## Installation & Running

1. **Requirements**

   - Python 3.x  
   - Tkinter *(bundled with Python)*  
   - Matplotlib → `pip install matplotlib`  
   - *(Optional but recommended)* Sun‑Valley TTK theme → `pip install sv_ttk`

2. **Files**

   Place the following files in the **same directory**:

   - `main.py` – entry point  
   - `gui.py` – main GUI class  
   - `neural_network.py` – `NeuralNetwork` class  
   - `utils.py` – mathematical helpers  
   - `gui_components.py` – GUI widgets (e.g. ToolTip)

3. **Run**

   Open a terminal/cmd, navigate to the project folder and execute:

   ```bash
   python main.py
   ```

---

## User Guide

The UI has two major areas: the **Control Panel** (left) and the **Visualisation & Results Panel** (right).

### 1. Network Configuration (Left Panel – Top)

Define your network architecture here.

| Control | Description |
|---------|-------------|
| **Switch Theme** | Toggle between light & dark mode (works if `sv_ttk` is installed). |
| **Load Network** | Load a previously saved network & training state (`.json`). |
| **Save Network** | Save the current network & training state (`.json`) *enabled after building the net*. |
| **# Hidden Layers** | Select the number of hidden layers (0‑100). Updates the *Hidden K.X Neurons/Actv.* fields below. |
| **# Input Neurons** | Number of input features. |
| **# Output Neurons** | Number of outputs (usually = number of classes in classification). |
| **Output Actv. Func.** | Activation function for the output layer (`sigmoid`, `softmax`, `linear`). |
| **Hidden K.X Neurons / Actv.** | Enter neuron count & activation for each hidden layer. |
| **Build & Draw Network** | Creates the neural net with the above settings and draws it on the right. Enables the training controls. |

### 2. Data & Training Parameters (Left Panel – Middle)

Set the data and training hyper‑parameters.

- **Load Data (CSV)**  
  Load training & target data from a CSV file.  
  - The first row may be a header (optional).  
  - Data must be numeric.  
  - First **N** columns → input **X**, next **M** columns → target **Y** (`N` = # Input Neurons, `M` = # Output Neurons).

- **Input Data (X)**  
  Enter input samples manually.  
  - **Format:** Each row = one sample. If you wish, separate multiple samples on one line with semicolon (`;`). Inside a sample, separate features by comma (`,`).  
  - **Example (2 samples, 2 features):**  
    ```
    0.1,0.5
    0.8,0.2
    ```  
    or  
    ```
    0.1,0.5;0.8,0.2
    ```

- **Target Outputs (Y)**  
  Enter target values manually. Same format as **X**.  
  - **Important (Classification):** If you choose `cross_entropy` loss with multiple classes (# Output > 1 & Output Actv. = `softmax`), **Y must be one‑hot**. E.g. for 3 classes and true class = 2nd: `0,1,0`. If a single label is provided, the program will try to convert it to one‑hot.

- **Loss Function** – `mean_squared_error` or `cross_entropy`  
- **Optimizer** – `sgd`, `momentum`, or `adam`  
- **# Epochs** – How many times the full dataset is fed through the net.  
- **Learning Rate** – Step size for weight updates.

### 3. Execution & Monitoring (Left Panel – Bottom)

Controls for running and observing the network.

| Button / Option | What it does |
|-----------------|--------------|
| **Detailed Fwd Step (?)** | If checked, *Forward Step* shows each weight·input + bias separately; otherwise a whole layer at once. |
| **Forward Step (1 sample)** | Runs a single forward‑prop step with the first X sample, advancing one calculation per click. |
| **Full Forward (1 sample)** | Runs complete forward‑prop once and shows the result. |
| **Backward Step (1 sample)** | Performs backward‑prop step‑by‑step using the first Y sample, after a full forward pass. |
| **Train Step‑by‑Step (start 1 sample)** | Runs one training step (forward + backward) on the first sample. Continue with **Next Step in Training →**. |
| **Current Phase** | Shows the current phase during step‑by‑step training (e.g. Forward, Backward). |
| **Next Step in Training →** | Moves to the next calculation step after *Train Step‑by‑Step* is started. |
| **Show Steps in Auto‑Train / Delay(s)** | If checked, auto‑training visualises each (sub‑)step. Set delay between steps (e.g. 0.05 s). |
| **Start Training (Auto)** | Trains automatically for the specified epochs. |
| **Progress Bar** | Shows epoch progress during auto‑training. |
| **Reset Simulation** | Resets everything (network, data, graphs, settings).

### 4. Visualisation & Results Panel (Right Side)

Organised in tabs.

- **Network Visualisation**  
  - Displays the built network graphically.  
  - Save image as **`.eps`** (convert later to PNG via e.g. Ghostscript).  
  - Toggle check‑boxes to show/hide weights, biases, neuron values.  
  - **Click a neuron** to open a detail window with its values.

- **Logs & Output**  
  - Shows operations, error messages, training progress, and key results.

- **Weights & Biases (Edit)**  
  - After building the net, edit weight matrices or bias vectors manually.  
  - Select the matrix/vector, change values, then click **Apply Changes** (optimizer state resets).

- **Loss Graph**  
  - Mean loss per epoch during auto‑training (Matplotlib toolbar enabled).

- **Accuracy Graph**  
  - For classification (Cross‑Entropy + Softmax), accuracy per epoch.

- **Confusion Matrix**  
  - For classification, a text‑based confusion matrix after training.  
  - Format `Real\Pred | C0 | C1 | ...`.

- **Metrics**  
  - Shows metrics during/after training.  
  - *Regression:* Mean Loss.  
  - *Classification:* Mean Loss, Accuracy, Precision (macro & per‑class), Recall, F1 (macro & per‑class).  
  - After *Full Forward*, shows predicted vs. real values for the last sample.

---

## Tips & Troubleshooting

- **# Hidden Layers**  
  With many layers (>10) or many neurons per layer (>50), config fields become scrollable.

- **Performance**  
  Training large nets or many epochs can be slow if *Show Steps in Auto‑Train* and especially *Detailed Fwd Step* are enabled. Disable them for faster runs.

- **Data Format**  
  Ensure numeric values and correct separators (`,` for features, `;` for samples).

- **Classification Metrics**  
  To view Accuracy, Precision, Recall, F1 and Confusion Matrix:  
  - Output neurons = number of classes  
  - Output activation = `softmax`  
  - Loss = `cross_entropy`  
  - Y targets must be **one‑hot** encoded.

Enjoy designing and exploring your own neural networks! 🚀
