# Deep Network Design Simulator

A desktop educational simulator for designing small dense neural networks and inspecting their calculations. Forward propagation, backpropagation, and optimizer updates use Python lists and manual mathematical operations. Matplotlib plots the training history, and Tkinter provides the interface. The application labels and logs are primarily Turkish.

## Setup and launch

Use Python 3.10 or newer with Tkinter and a graphical desktop. Python 3.13 with Tcl/Tk 8.6 was used for local validation. On Windows, select Tcl/Tk support when installing Python. On Linux, install your distribution's Tkinter package if `python -m tkinter` is unavailable. A headless terminal alone cannot display the application.

From the repository root, create an isolated environment and install the runtime dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe source_code/main.py
```

On macOS or Linux, use the environment's `bin/python`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python source_code/main.py
```

The equivalent module launch from the repository root is `python -m source_code.main`, using the Python interpreter from your environment. Keep the source files together in `source_code`; there is no `main.py` at the repository root.

The optional Sun Valley theme enables light/dark switching:

```sh
python -m pip install sv_ttk
```

Run that command with your environment's interpreter. Without `sv_ttk`, the simulator uses the default Tk theme.

`Simulator.exe` is an existing Windows binary committed to this repository. It is not rebuilt by these setup commands and does not incorporate subsequent source changes. The source application is the maintained and tested launch path; the binary has not been validated against it.

## Design and inspect a network

The left panel contains architecture, data, and training controls. The right panel contains visualization, logs, editable weights and biases, and results.

1. Set the input feature count, number of hidden layers, neuron counts, and output count.
2. Choose each layer's activation: sigmoid, ReLU, tanh, linear, or softmax.
3. Select **Ağı Kur ve Çiz** (Build and Draw Network). A network can have zero hidden layers. All input and layer sizes must be positive integers.
4. Supply input samples and targets. Use the forward controls to inspect one sample, then the backward or training controls to update weights.

The network graph can show activations, weighted sums, biases, and connection weights. Clicking a neuron shows its details. The weight/bias editor applies manual values and resets optimizer state.

Training supports SGD, momentum, and Adam. Step-by-step controls show individual forward and backward calculations. Automatic training iterates the dataset for the selected epochs; showing individual steps and adding a delay makes it slower. Start with a small network and a small epoch count because training runs on the GUI thread.

## Data format

For manual data, separate features with commas. Separate samples with newlines, semicolons, or both. For example, two samples with two features:

```text
0.1,0.5
0.8,0.2
```

The equivalent single-line input is `0.1,0.5;0.8,0.2`. Provide one matching target sample for each input sample. The individual forward/backward controls use the first sample.

CSV input can have a text header or no header. Each numeric row contains the input features first, followed by target values. For two input features and one output:

```csv
feature1,feature2,target
0.1,0.5,0
0.8,0.2,1
```

The first numeric row is preserved when there is no header. Invalid rows are skipped with a log message. The text boxes display up to five imported samples; automatic training uses all successfully imported samples, while the individual step controls use the first displayed sample. Automatic training continues to use the imported dataset until the simulation is reset or another CSV is loaded, even if the preview text is edited.

Use `mean_squared_error` for regression. It reports one half of the mean squared error across output neurons. For classification, use `cross_entropy` with a **softmax output layer**. Other output activations are rejected during cross-entropy backpropagation. Set the output count to the number of classes and supply one-hot targets such as `0,1,0`, or a zero-based class index such as `1`. The simulator converts class indices to one-hot vectors.

Accuracy, precision, recall, F1, and the text confusion matrix are available for cross-entropy with softmax. These are results from the provided samples, not independently validated model benchmarks.

## Save and load

Save/load controls use JSON containing the architecture, weights, biases, loss choice, optimizer state, and training history. Training data and the full set of training settings are not stored in that file. Supplied weight matrices and bias vectors must include every layer, match its dimensions, and contain finite numeric values. Invalid model construction leaves the previous network intact.

Save the graph as Encapsulated PostScript (`.eps`) using the image save control. Converting it to another format requires a separate EPS-capable tool.

## Development checks

Install the development requirements using your environment's interpreter:

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m bandit -r source_code
python -m pip_audit
```

Tests exercise network construction, state reset, deterministic forward calculations, numerical gradient comparisons, multiline/CSV parsing, GUI construction, and tooltip behavior. GUI tests need a display and skip when Tk cannot create a window; check the reported skip count before treating a headless run as GUI evidence. Tests use tiny in-memory networks and temporary CSV fixtures without external models, datasets, or expensive training.

The repository currently has no GitHub Actions workflow. Local checks do not validate the prebuilt executable, every interactive control, long training runs, or cross-platform desktop behavior.

## License

[MIT](LICENSE), copyright 2025 Tuğcan Topaloğlu.
