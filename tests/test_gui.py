import copy
import importlib
import json
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

from source_code.gui import DeepLearningSimulatorGUI
from source_code.gui_components import ToolTip


class GuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.addCleanup(self._close_root)
        self.errors = []
        self.root.report_callback_exception = lambda *args: self.errors.append(args)
        self.app = DeepLearningSimulatorGUI(self.root)
        self.app.initial_draw()

    def _close_root(self):
        for callback in self.root.tk.call("after", "info"):
            self.root.after_cancel(callback)
        self.root.destroy()

    def test_gui_constructs_and_builds_default_network(self):
        self.app.build_and_draw_network()
        self.root.update()
        self.assertEqual(self.errors, [])
        self.assertEqual(self.app.network.layer_configs, [(3, "relu"), (1, "sigmoid")])
        self.assertEqual(len(self.app.network.weights[0]), 2)
        self.assertEqual(len(self.app.network.weights[1]), 3)
        self.assertGreater(len(self.app.canvas.find_all()), 0)

    def test_module_entry_import_does_not_open_window(self):
        with patch("tkinter.Tk", side_effect=AssertionError("Import opened a window")):
            importlib.import_module("source_code.main")

    def test_manual_data_accepts_newlines_and_semicolons(self):
        self.assertEqual(
            self.app._parse_input_data("0.1,0.5\n\n0.8,0.2;1,2\n", 2),
            [[0.1, 0.5], [0.8, 0.2], [1.0, 2.0]],
        )
        self.app.x_input_text.delete("1.0", tk.END)
        self.app.x_input_text.insert("1.0", "0.1,0.5\n0.8,0.2")
        self.app.y_input_text.delete("1.0", tk.END)
        self.app.y_input_text.insert("1.0", "0\n1")
        self.assertEqual(self.app._get_first_training_sample_for_step_ops(), ([0.1, 0.5], [0.0]))

    def test_multiline_class_indices_convert_to_one_hot(self):
        self.app.loss_function_var.set("cross_entropy")
        self.assertEqual(self.app._parse_input_data("1\n0", 3, True, 3), [[0, 1, 0], [1, 0, 0]])

    def test_csv_preserves_all_samples_with_and_without_header(self):
        with tempfile.TemporaryDirectory() as directory:
            for header in ("", "feature1,feature2,target\n"):
                with self.subTest(header=header):
                    csv_path = Path(directory) / "samples.csv"
                    csv_path.write_text(header + "0.1,0.5,0\n0.8,0.2,1\n", encoding="utf-8")
                    with patch("source_code.gui.filedialog.askopenfilename", return_value=str(csv_path)):
                        self.app.load_data_from_csv()
                    self.assertEqual(self.app.training_data_X, [[0.1, 0.5], [0.8, 0.2]])
                    self.assertEqual(self.app.training_data_Y, [[0.0], [1.0]])

    def test_checkbutton_tooltip_reuses_one_window_and_closes(self):
        widget = ttk.Checkbutton(self.root, text="Details")
        widget.pack()
        tooltip = ToolTip(widget, "Show detailed steps")
        tooltip.show_tooltip()
        window = tooltip.tooltip_window
        self.assertTrue(window.winfo_exists())
        tooltip.show_tooltip()
        self.assertIs(tooltip.tooltip_window, window)
        tooltip.hide_tooltip()
        self.assertIsNone(tooltip.tooltip_window)

    def test_invalid_saved_model_does_not_replace_network_or_optimizer(self):
        self.app.build_and_draw_network()
        before = copy.deepcopy(self.app.network.__dict__)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps({
                "input_size": 2,
                "layer_configs_full": [[2, "linear"]],
                "weights": [[[1, 2], [3]]],
                "biases": [[0, 0]],
                "optimizer_state": {"adam_t": 999},
            }), encoding="utf-8")
            with patch("source_code.gui.filedialog.askopenfilename", return_value=str(path)), patch("source_code.gui.messagebox.showerror") as showerror:
                self.app.load_network_with_state()
            showerror.assert_called_once()
        self.assertEqual(self.app.network.__dict__, before)


if __name__ == "__main__":
    unittest.main()
