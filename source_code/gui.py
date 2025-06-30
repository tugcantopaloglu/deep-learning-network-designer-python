# Arayüz elemanlarının oluşturulması, olayların yönetilmesi ve sinir ağı işlemleriyle
# etkileşim bu sınıf üzerinden yürütülür.

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext, filedialog
import math
import random
import json
import csv
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from utils import ACTIVATION_FUNCTIONS, LOSS_FUNCTIONS
from neural_network import NeuralNetwork
from gui_components import ToolTip

try:
    import sv_ttk
except ImportError:
    sv_ttk = None

class DeepLearningSimulatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Derin Ağ Tasarım Simülatörü - Tuğcan Topaloğlu")
        master.geometry("1600x1000") 
        self.current_theme = "light" 
        if sv_ttk: sv_ttk.set_theme(self.current_theme) 
        self.network = NeuralNetwork()
        self.forward_pass_gen, self.backward_pass_gen = None, None
        self.training_data_X, self.training_data_Y = [], []
        self.current_training_sample_idx, self.is_training_step_by_step_active = 0, False
        self.current_epoch_losses, self.current_epoch_accuracies = [], []
        self.current_training_X_sample, self.current_training_Y_sample = None, None 
        self.detailed_forward_steps = tk.BooleanVar(value=False) 
        self.auto_train_watch_steps_var = tk.BooleanVar(value=False) 
        self.auto_train_step_delay_var = tk.DoubleVar(value=0.05) 
        self.show_weights_on_canvas_var = tk.BooleanVar(value=True)
        self.show_biases_on_canvas_var = tk.BooleanVar(value=True)
        self.show_neuron_values_on_canvas_var = tk.BooleanVar(value=True)
        self.show_weights_on_canvas_var.trace_add("write", lambda *args: self.draw_network_on_canvas())
        self.show_biases_on_canvas_var.trace_add("write", lambda *args: self.draw_network_on_canvas())
        self.show_neuron_values_on_canvas_var.trace_add("write", lambda *args: self.draw_network_on_canvas())

        self.style = ttk.Style()
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        left_scrollbar_frame = ttk.Frame(self.main_frame)
        left_scrollbar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        bg_color = self.style.lookup('TFrame', 'background')
        if not bg_color: bg_color = "SystemButtonFace" if self.master.tk.call('tk', 'windowingsystem') == 'win32' else "white"
        
        self.left_canvas = tk.Canvas(left_scrollbar_frame, borderwidth=0, background=bg_color, highlightthickness=0)
        left_v_scroll = ttk.Scrollbar(left_scrollbar_frame, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_v_scroll.set)
        left_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.left_frame_content = ttk.Frame(self.left_canvas) 
        self.left_canvas_window_id = self.left_canvas.create_window((0,0), window=self.left_frame_content, anchor="nw")
        
        self.left_frame_content.bind("<Configure>", self._on_left_content_configure)
        self.left_canvas.bind("<Configure>", self._on_left_canvas_configure)
        
        right_frame = ttk.Frame(self.main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._setup_left_panel_controls()
        self._setup_right_panel_visualization_and_logs(right_frame)
        
        self.log_message("Simülatör başlatıldı. Tema: " + (self.current_theme if sv_ttk else "Varsayılan"))
        self.log_message("İşlemleri ve sonuçları bu panelden ve grafik sekmelerinden canlı olarak izleyebilirsiniz.")
        self.master.after(150, self.initial_draw)

    def _on_left_content_configure(self, event):
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _on_left_canvas_configure(self, event):
        canvas_width = event.width
        self.left_canvas.itemconfig(self.left_canvas_window_id, width=canvas_width)
        self.left_frame_content.config(width=canvas_width) 

    def _setup_left_panel_controls(self):
        parent = self.left_frame_content
        controls_panel = ttk.LabelFrame(parent, text="Ağ Yapılandırması ve Kontroller", padding="10")
        controls_panel.pack(fill=tk.X, pady=5, expand=False)
        
        theme_button = ttk.Button(controls_panel, text="Tema Değiştir", command=self.toggle_theme)
        theme_button.grid(row=0, column=0, columnspan=1, pady=(0,10), sticky=tk.W)
        
        file_ops_frame = ttk.Frame(controls_panel)
        file_ops_frame.grid(row=0, column=1, columnspan=1, pady=(0,10), sticky=tk.EW)
        self.load_network_button = ttk.Button(file_ops_frame, text="Ağı Yükle", command=self.load_network_with_state)
        self.load_network_button.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        self.save_network_button = ttk.Button(file_ops_frame, text="Ağı Kaydet", command=self.save_network_with_state, state=tk.DISABLED)
        self.save_network_button.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        ttk.Label(controls_panel, text="Gizli Katman Sayısı:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.num_hidden_layers_var = tk.IntVar(value=1)
        self.num_hidden_layers_spinbox = ttk.Spinbox(controls_panel, from_=0, to=100, textvariable=self.num_hidden_layers_var, width=5, command=self.update_layer_config_entries)
        self.num_hidden_layers_spinbox.grid(row=1, column=1, sticky=tk.EW, pady=2)
        ttk.Label(controls_panel, text="Giriş Nöron Sayısı:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.input_size_var = tk.IntVar(value=2)
        ttk.Entry(controls_panel, textvariable=self.input_size_var, width=7).grid(row=2, column=1, sticky=tk.EW, pady=2)
        ttk.Label(controls_panel, text="Çıkış Nöron Sayısı:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.output_size_var = tk.IntVar(value=1)
        ttk.Entry(controls_panel, textvariable=self.output_size_var, width=7).grid(row=3, column=1, sticky=tk.EW, pady=2)
        ttk.Label(controls_panel, text="Çıkış Aktv. Fonk:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.output_activation_var = tk.StringVar(value="sigmoid")
        self.output_activation_combo = ttk.Combobox(controls_panel, textvariable=self.output_activation_var, values=list(ACTIVATION_FUNCTIONS.keys()), state="readonly", width=10)
        self.output_activation_combo.grid(row=4, column=1, sticky=tk.EW, pady=2)
        self.layer_config_frame = ttk.Frame(controls_panel)
        self.layer_config_frame.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=5)
        self.layer_entries = [] 
        self.update_layer_config_entries() 
        self.build_network_button = ttk.Button(controls_panel, text="Ağı Kur ve Çiz", command=self.build_and_draw_network)
        self.build_network_button.grid(row=6, column=0, columnspan=2, pady=10, sticky=tk.EW)

        data_panel = ttk.LabelFrame(parent, text="Veri ve Eğitim Parametreleri", padding="10")
        data_panel.pack(fill=tk.X, pady=5, expand=False)
        self.load_csv_button = ttk.Button(data_panel, text="Veri Yükle (CSV)", command=self.load_data_from_csv)
        self.load_csv_button.grid(row=0, column=0, columnspan=2, pady=(0,5), sticky=tk.EW)
        input_format_label_text = "Giriş Verileri (X):\nVirgül ile ayırarak input verilerini girebilirsiniz.\nÖrn: 0.1,0.2 ..."
        ttk.Label(data_panel, text=input_format_label_text, justify=tk.LEFT).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        self.x_input_text = scrolledtext.ScrolledText(data_panel, height=3, width=30, font=('Monospace', 9))
        self.x_input_text.insert(tk.END, "0.05,0.1"); self.x_input_text.grid(row=2, column=0, columnspan=2, pady=2, sticky=tk.EW)
        ttk.Label(data_panel, text="Hedef Çıktılar (Y): Format X ile aynı.").grid(row=3, column=0, columnspan=2, sticky=tk.W)
        self.y_input_text = scrolledtext.ScrolledText(data_panel, height=3, width=30, font=('Monospace', 9))
        self.y_input_text.insert(tk.END, "0.01"); self.y_input_text.grid(row=4, column=0, columnspan=2, pady=2, sticky=tk.EW)
        ttk.Label(data_panel, text="Kayıp Fonksiyonu:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.loss_function_var = tk.StringVar(value="mean_squared_error")
        self.loss_function_combo = ttk.Combobox(data_panel, textvariable=self.loss_function_var, values=list(LOSS_FUNCTIONS.keys()), state="readonly", width=15)
        self.loss_function_combo.grid(row=5, column=1, sticky=tk.EW, pady=2)
        self.loss_function_combo.bind("<<ComboboxSelected>>", self.on_loss_function_change)
        ttk.Label(data_panel, text="Optimizasyon:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.optimizer_var = tk.StringVar(value="sgd")
        self.optimizer_combo = ttk.Combobox(data_panel, textvariable=self.optimizer_var, values=["sgd", "momentum", "adam"], state="readonly", width=10)
        self.optimizer_combo.grid(row=6, column=1, sticky=tk.EW, pady=2)
        ttk.Label(data_panel, text="Epoch Sayısı:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Entry(data_panel, textvariable=self.epochs_var, width=7).grid(row=7, column=1, sticky=tk.EW, pady=2)
        ttk.Label(data_panel, text="Öğrenme Oranı:").grid(row=8, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Entry(data_panel, textvariable=self.lr_var, width=7).grid(row=8, column=1, sticky=tk.EW, pady=2)

        run_panel = ttk.LabelFrame(parent, text="Çalıştırma ve İzleme", padding="10")
        run_panel.pack(fill=tk.BOTH, pady=5, expand=True)
        detail_fw_frame = ttk.Frame(run_panel); detail_fw_frame.pack(fill=tk.X, pady=1)
        cb_detail = ttk.Checkbutton(detail_fw_frame, text="Detaylı İleri Adım", variable=self.detailed_forward_steps)
        cb_detail.pack(side=tk.LEFT, padx=(0,5))
        ToolTip(cb_detail, "Seçili ise, ileri yayılımın her bir ağırlık çarpımı ve bias toplama adımını ayrı ayrı gösterir.\nSeçili değilse, her katmanın sonucunu tek adımda gösterir.")
        
        self.forward_step_button = ttk.Button(run_panel, text="İleri Adım (1 Örnek)", command=self.execute_forward_step, state=tk.DISABLED); self.forward_step_button.pack(fill=tk.X, pady=2)
        self.forward_all_button = ttk.Button(run_panel, text="Tüm İleri Yayılım (1 Örnek)", command=self.execute_forward_all, state=tk.DISABLED); self.forward_all_button.pack(fill=tk.X, pady=2)
        self.backward_step_button = ttk.Button(run_panel, text="Geri Adım (1 Örnek)", command=self.execute_backward_step, state=tk.DISABLED); self.backward_step_button.pack(fill=tk.X, pady=2)
        ttk.Separator(run_panel, orient='horizontal').pack(fill='x', pady=5)
        self.train_step_by_step_button = ttk.Button(run_panel, text="Adım Adım Eğit (1 Örnek Başlat)", command=self.start_step_by_step_training_one_sample, state=tk.DISABLED); self.train_step_by_step_button.pack(fill=tk.X, pady=2)
        self.current_training_phase_label = ttk.Label(run_panel, text="Mevcut Aşama: -"); self.current_training_phase_label.pack(fill=tk.X, pady=(2,0))
        self.train_next_step_button = ttk.Button(run_panel, text="Eğitimde Sonraki Adım >>", command=self.execute_next_training_step, state=tk.DISABLED); self.train_next_step_button.pack(fill=tk.X, pady=2)
        ttk.Separator(run_panel, orient='horizontal').pack(fill='x', pady=5)
        auto_train_options_frame = ttk.Frame(run_panel); auto_train_options_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(auto_train_options_frame, text="Oto. Eğitimde Adımları İzle", variable=self.auto_train_watch_steps_var).pack(side=tk.LEFT, padx=(0,5))
        ttk.Label(auto_train_options_frame, text="Gecikme(s):").pack(side=tk.LEFT)
        ttk.Entry(auto_train_options_frame, textvariable=self.auto_train_step_delay_var, width=5).pack(side=tk.LEFT)
        self.train_button = ttk.Button(run_panel, text="Eğitimi Başlat (Otomatik)", command=self.start_training_auto, state=tk.DISABLED); self.train_button.pack(fill=tk.X, pady=2)
        self.progress_bar = ttk.Progressbar(run_panel, orient="horizontal", mode="determinate", length=200)
        self.progress_bar.pack(fill=tk.X, pady=(5,2))
        self.reset_button = ttk.Button(run_panel, text="Simülasyonu Sıfırla", command=self.reset_simulation); self.reset_button.pack(fill=tk.X, pady=(5,2))

    def _setup_right_panel_visualization_and_logs(self, parent_frame):
        vis_log_notebook = ttk.Notebook(parent_frame)
        vis_log_notebook.pack(fill=tk.BOTH, expand=True)
        self.vis_frame = ttk.Frame(vis_log_notebook); vis_log_notebook.add(self.vis_frame, text='Ağ Görselleştirmesi')
        vis_toolbar_frame = ttk.Frame(self.vis_frame); vis_toolbar_frame.pack(fill=tk.X, pady=2)
        self.save_canvas_button = ttk.Button(vis_toolbar_frame, text="Görseli Kaydet (.eps)", command=self.save_canvas_as_eps, state=tk.DISABLED)
        self.save_canvas_button.pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(vis_toolbar_frame, text="Ağırlıklar", variable=self.show_weights_on_canvas_var).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(vis_toolbar_frame, text="Biaslar", variable=self.show_biases_on_canvas_var).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(vis_toolbar_frame, text="Nöron Değerleri", variable=self.show_neuron_values_on_canvas_var).pack(side=tk.LEFT, padx=2)
        self.canvas = tk.Canvas(self.vis_frame, bg='white', scrollregion=(0,0,1200,800)) 
        hbar = ttk.Scrollbar(self.vis_frame, orient=tk.HORIZONTAL, command=self.canvas.xview); hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar = ttk.Scrollbar(self.vis_frame, orient=tk.VERTICAL, command=self.canvas.yview); vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set); self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.neuron_canvas_objects, self.neuron_value_texts, self.neuron_z_value_texts = {}, {}, {}
        self.connection_canvas_objects, self.connection_weight_value_texts, self.neuron_bias_value_texts = {}, {}, {}
        self.log_frame = ttk.Frame(vis_log_notebook); vis_log_notebook.add(self.log_frame, text='Loglar ve Sonuçlar')
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=10, state=tk.DISABLED, font=('Monospace', 9), wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.weights_biases_editor_frame = ttk.Frame(vis_log_notebook); vis_log_notebook.add(self.weights_biases_editor_frame, text='Ağırlıklar & Biaslar (Düzenle)')
        self._setup_weights_biases_editor_tab() 
        self.loss_graph_tab_frame = ttk.Frame(vis_log_notebook); vis_log_notebook.add(self.loss_graph_tab_frame, text='Kayıp Grafiği')
        self.fig_loss = Figure(figsize=(5, 3.5), dpi=100) 
        self.ax_loss = self.fig_loss.add_subplot(111)
        self.loss_canvas_widget = FigureCanvasTkAgg(self.fig_loss, master=self.loss_graph_tab_frame) 
        toolbar_loss = NavigationToolbar2Tk(self.loss_canvas_widget, self.loss_graph_tab_frame); toolbar_loss.update()
        self.loss_canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True); self.update_loss_graph() 
        self.accuracy_graph_tab_frame = ttk.Frame(vis_log_notebook); vis_log_notebook.add(self.accuracy_graph_tab_frame, text='Doğruluk Grafiği')
        self.fig_accuracy = Figure(figsize=(5, 3.5), dpi=100)
        self.ax_accuracy = self.fig_accuracy.add_subplot(111)
        self.accuracy_canvas_widget = FigureCanvasTkAgg(self.fig_accuracy, master=self.accuracy_graph_tab_frame)
        toolbar_acc = NavigationToolbar2Tk(self.accuracy_canvas_widget, self.accuracy_graph_tab_frame); toolbar_acc.update()
        self.accuracy_canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True); self.update_accuracy_graph()
        self.confusion_matrix_tab_frame = ttk.Frame(vis_log_notebook); vis_log_notebook.add(self.confusion_matrix_tab_frame, text='Karmaşıklık Matrisi')
        self.cm_text_area = scrolledtext.ScrolledText(self.confusion_matrix_tab_frame, height=10, state=tk.DISABLED, font=('Monospace', 10), wrap=tk.NONE)
        self.cm_text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.metrics_frame = ttk.Frame(vis_log_notebook); vis_log_notebook.add(self.metrics_frame, text='Metrikler')
        self.metrics_text = scrolledtext.ScrolledText(self.metrics_frame, height=10, state=tk.DISABLED, font=('Monospace', 9))
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def toggle_theme(self):
        if not sv_ttk: messagebox.showinfo("Tema", "Sun-Valley teması yüklü değil.", parent=self.master); return
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        sv_ttk.set_theme(self.current_theme)
        self.log_message(f"Tema değiştirildi: {self.current_theme}")
        bg_color = self.style.lookup('TFrame', 'background')
        if hasattr(self, 'wb_canvas'): self.wb_canvas.config(background=bg_color)
        if hasattr(self, 'left_canvas'): self.left_canvas.config(background=bg_color)
        if hasattr(self, 'canvas'): self.canvas.config(bg="white" if self.current_theme == "light" else "#333333")
        self.draw_network_on_canvas()

    def _get_layer_display_name(self, layer_config_idx):
        if not self.network or not self.network.layer_configs or layer_config_idx < 0 or layer_config_idx >= len(self.network.layer_configs): return f"Bilinmeyen L{layer_config_idx}"
        is_output_layer = (layer_config_idx == len(self.network.layer_configs) - 1)
        if is_output_layer: return "Çıkış Katmanı"
        else: return f"Gizli Katman {layer_config_idx + 1}"

    def _get_neuron_display_name(self, layer_type_str, neuron_idx_0_based, layer_number_1_based_if_hidden=None):
        if layer_type_str == "Giriş": return f"Giriş N{neuron_idx_0_based + 1}"
        elif layer_type_str == "Gizli": return f"Gizli L{layer_number_1_based_if_hidden} N{neuron_idx_0_based + 1}"
        elif layer_type_str == "Çıkış": return f"Çıkış N{neuron_idx_0_based + 1}"
        return f"N{neuron_idx_0_based + 1}"

    def _get_source_layer_display_name_for_weights(self, weight_matrix_idx):
        if weight_matrix_idx == 0: return "Giriş Katmanı"
        else: return self._get_layer_display_name(weight_matrix_idx - 1)

    def _setup_weights_biases_editor_tab(self):
        controls_frame = ttk.Frame(self.weights_biases_editor_frame); controls_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(controls_frame, text="Düzenlenecek:").pack(side=tk.LEFT, padx=(0,5))
        self.wb_layer_choice_var = tk.StringVar()
        self.wb_choice_combo = ttk.Combobox(controls_frame, textvariable=self.wb_layer_choice_var, state="readonly", width=45)
        self.wb_choice_combo.pack(side=tk.LEFT, expand=True, fill=tk.X); self.wb_choice_combo.bind("<<ComboboxSelected>>", self._on_wb_layer_selected)
        self.wb_editor_grid_outer_frame = ttk.Frame(self.weights_biases_editor_frame); self.wb_editor_grid_outer_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        self.wb_canvas = tk.Canvas(self.wb_editor_grid_outer_frame, borderwidth=0, background=self.style.lookup('TFrame', 'background'))
        self.wb_editor_grid_frame = ttk.Frame(self.wb_canvas) 
        self.wb_v_scroll = ttk.Scrollbar(self.wb_editor_grid_outer_frame, orient="vertical", command=self.wb_canvas.yview)
        self.wb_h_scroll = ttk.Scrollbar(self.wb_editor_grid_outer_frame, orient="horizontal", command=self.wb_canvas.xview)
        self.wb_canvas.configure(yscrollcommand=self.wb_v_scroll.set, xscrollcommand=self.wb_h_scroll.set)
        self.wb_v_scroll.pack(side="right", fill="y"); self.wb_h_scroll.pack(side="bottom", fill="x"); self.wb_canvas.pack(side="left", fill="both", expand=True)
        self.wb_canvas_window = self.wb_canvas.create_window((4,4), window=self.wb_editor_grid_frame, anchor="nw")
        self.wb_editor_grid_frame.bind("<Configure>", lambda e: self.wb_canvas.configure(scrollregion=self.wb_canvas.bbox("all")))
        self.wb_canvas.bind("<Configure>", self._on_wb_canvas_configure)
        self.wb_apply_button = ttk.Button(self.weights_biases_editor_frame, text="Değişiklikleri Ağa Uygula", command=self._apply_weights_biases_from_editor, state=tk.DISABLED)
        self.wb_apply_button.pack(fill=tk.X, pady=(5,10), padx=5); self.wb_entry_vars = [] 

    def _on_wb_canvas_configure(self, event):
        self.wb_canvas.itemconfig(self.wb_canvas_window, width=event.width -8); self.wb_editor_grid_frame.update_idletasks(); self.wb_canvas.config(scrollregion=self.wb_canvas.bbox("all"))

    def _on_wb_layer_selected(self, event=None):
        self._populate_weights_biases_editor(); self.wb_apply_button.config(state=tk.NORMAL if self.wb_layer_choice_var.get() else tk.DISABLED)

    def _populate_wb_combo(self):
        choices = []
        if self.network and self.network.weights:
            for i in range(len(self.network.weights)):
                source_name, target_name = self._get_source_layer_display_name_for_weights(i), self._get_layer_display_name(i) 
                num_s, num_t = self.input_size_var.get() if i == 0 else self.network.layer_configs[i-1][0], self.network.layer_configs[i][0]
                choices.append(f"Ağırlıklar: {source_name} ({num_s}N) → {target_name} ({num_t}N) [W{i}]")
            for i in range(len(self.network.biases)):
                layer_name, num_n = self._get_layer_display_name(i), self.network.layer_configs[i][0]
                choices.append(f"Biaslar: {layer_name} ({num_n}N) [B{i}]")
        current_selection = self.wb_layer_choice_var.get(); self.wb_choice_combo['values'] = choices
        if choices:
            if current_selection in choices: self.wb_layer_choice_var.set(current_selection)
            else: self.wb_layer_choice_var.set(choices[0])
            self._populate_weights_biases_editor(); self.wb_apply_button.config(state=tk.NORMAL)
        else: self.wb_layer_choice_var.set(""); self._clear_wb_editor_grid(); self.wb_apply_button.config(state=tk.DISABLED)

    def _clear_wb_editor_grid(self):
        for widget in self.wb_editor_grid_frame.winfo_children(): widget.destroy()
        self.wb_entry_vars = []

    def _populate_weights_biases_editor(self):
        self._clear_wb_editor_grid()
        if not self.network or not self.network.weights: return
        selection_str = self.wb_layer_choice_var.get()
        if not selection_str: return
        entry_width, font_size = 10, 8
        if selection_str.startswith("Ağırlıklar:"):
            try: w_matrix_idx = int(selection_str.split("[W")[1].split("]")[0])
            except: self.log_message(f"Hata: Ağırlık matrisi indeksi okunamadı: {selection_str}", True); return
            if w_matrix_idx >= len(self.network.weights): self.log_message(f"Hata: Ağırlık matrisi indeksi {w_matrix_idx} sınırların dışında.", True); return
            weights_matrix = self.network.weights[w_matrix_idx]; num_prev_n, num_curr_n = len(weights_matrix), len(weights_matrix[0]) if weights_matrix else 0
            target_layer_name_full = self._get_layer_display_name(w_matrix_idx)
            target_layer_type_str, target_layer_num_1_based = ("Çıkış" if "Çıkış" in target_layer_name_full else "Gizli"), (int(target_layer_name_full.split(" ")[-1]) if "Gizli" in target_layer_name_full else None)
            ttk.Label(self.wb_editor_grid_frame, text="Kaynak N↓ / Hedef N→", font=('Calibri', font_size, 'italic'), anchor="w").grid(row=0, column=0, padx=2, pady=2, sticky="w")
            for j in range(num_curr_n): ttk.Label(self.wb_editor_grid_frame, text=self._get_neuron_display_name(target_layer_type_str, j, target_layer_num_1_based), width=entry_width+2, anchor="center", font=('Calibri', font_size, 'bold')).grid(row=0, column=j+1, padx=1, pady=2, sticky="ew")
            current_row_vars = []
            for r_idx in range(num_prev_n): 
                source_layer_name_full = self._get_source_layer_display_name_for_weights(w_matrix_idx)
                _source_layer_type = "Giriş" if "Giriş" in source_layer_name_full else "Gizli"
                _source_layer_num = int(source_layer_name_full.split(" ")[-1]) if "Gizli" in source_layer_name_full else None
                ttk.Label(self.wb_editor_grid_frame, text=self._get_neuron_display_name(_source_layer_type, r_idx, _source_layer_num), font=('Calibri', font_size, 'bold'), anchor="e").grid(row=r_idx+1, column=0, padx=2, pady=1, sticky="e")
                row_vars = []
                for c_idx in range(num_curr_n): 
                    entry_var = tk.DoubleVar(value=round(weights_matrix[r_idx][c_idx], 7))
                    ttk.Entry(self.wb_editor_grid_frame, textvariable=entry_var, width=entry_width, justify='right', font=('Calibri', font_size)).grid(row=r_idx+1, column=c_idx+1, padx=1, pady=1, sticky="ew")
                    row_vars.append(entry_var)
                current_row_vars.append(row_vars)
            self.wb_entry_vars = current_row_vars 
        elif selection_str.startswith("Biaslar:"):
            try: bias_vec_idx = int(selection_str.split("[B")[1].split("]")[0]) 
            except: self.log_message(f"Hata: Bias vektör indeksi okunamadı: {selection_str}", True); return
            if bias_vec_idx >= len(self.network.biases): self.log_message(f"Hata: Bias vektör indeksi {bias_vec_idx} sınırların dışında.", True); return
            biases_vector, num_neurons = self.network.biases[bias_vec_idx], len(self.network.biases[bias_vec_idx])
            layer_name_full = self._get_layer_display_name(bias_vec_idx)
            layer_type_str, layer_num_1_based = ("Çıkış" if "Çıkış" in layer_name_full else "Gizli"), (int(layer_name_full.split(" ")[-1]) if "Gizli" in layer_name_full else None)
            bias_vars_list = []
            for n_idx in range(num_neurons):
                ttk.Label(self.wb_editor_grid_frame, text=f"{self._get_neuron_display_name(layer_type_str, n_idx, layer_num_1_based)} Biası", font=('Calibri', font_size, 'bold'), anchor="e").grid(row=n_idx, column=0, padx=2, pady=2, sticky="e")
                entry_var = tk.DoubleVar(value=round(biases_vector[n_idx], 7))
                ttk.Entry(self.wb_editor_grid_frame, textvariable=entry_var, width=entry_width, justify='right', font=('Calibri', font_size)).grid(row=n_idx, column=1, padx=2, pady=2, sticky="ew")
                bias_vars_list.append(entry_var)
            self.wb_entry_vars = [bias_vars_list] 
        self.wb_editor_grid_frame.update_idletasks(); self.wb_canvas.config(scrollregion=self.wb_canvas.bbox("all"))
            
    def _apply_weights_biases_from_editor(self):
        if not self.network or not self.network.weights: messagebox.showerror("Hata", "Önce bir ağ kurun.", parent=self.master); return
        if not self.wb_entry_vars: messagebox.showinfo("Bilgi", "Düzenlenecek değer yok.", parent=self.master); return
        selection_str = self.wb_layer_choice_var.get()
        if not selection_str: return
        try:
            if selection_str.startswith("Ağırlıklar:"):
                w_idx = int(selection_str.split("[W")[1].split("]")[0])
                new_W = [[var.get() for var in r] for r in self.wb_entry_vars]
                if len(new_W) != len(self.network.weights[w_idx]) or (new_W and len(new_W[0]) != len(self.network.weights[w_idx][0])): raise ValueError("Okunan ağırlık matrisi boyutları ağdakiyle uyuşmuyor.")
                self.network.weights[w_idx] = new_W
                self.log_message(f"Ağırlıklar ({self._get_source_layer_display_name_for_weights(w_idx)} → {self._get_layer_display_name(w_idx)}) güncellendi.")
            elif selection_str.startswith("Biaslar:"):
                b_idx = int(selection_str.split("[B")[1].split("]")[0])
                new_B = [var.get() for var in self.wb_entry_vars[0]]
                if len(new_B) != len(self.network.biases[b_idx]): raise ValueError("Okunan bias vektörü boyutu ağdakiyle uyuşmuyor.")
                self.network.biases[b_idx] = new_B
                self.log_message(f"Biaslar ({self._get_layer_display_name(b_idx)}) güncellendi.")
            self.network.velocity_W,self.network.velocity_b,self.network.m_W,self.network.v_W,self.network.m_b,self.network.v_b,self.network.adam_t = [],[],[],[],[],[],0
            prev_n = self.input_size_var.get()
            for i in range(len(self.network.weights)): num_curr = len(self.network.biases[i]); self.network._initialize_optimizer_params(prev_n, num_curr); prev_n = num_curr
            self.draw_network_on_canvas(); self.current_epoch_losses, self.current_epoch_accuracies = [], []; self.update_loss_graph(); self.update_accuracy_graph(); self.update_metrics_display({})
            messagebox.showinfo("Başarılı", "Değişiklikler ağa uygulandı ve optimizer sıfırlandı.", parent=self.master)
        except ValueError as e: messagebox.showerror("Değer Hatası", f"Geçersiz değer girildi: {e}\nLütfen sayısal değerler girin.", parent=self.master)
        except Exception as e: messagebox.showerror("Hata", f"Uygulama sırasında hata: {e}", parent=self.master); import traceback; traceback.print_exc()
            
    def initial_draw(self):
        self.canvas.update_idletasks(); self.draw_network_on_canvas()

    def on_loss_function_change(self, event=None):
        selected_loss = self.loss_function_var.get()
        try: 
            self.network.set_loss_function(selected_loss); self.log_message(f"Kayıp fonksiyonu değiştirildi: {selected_loss}")
            if selected_loss == "cross_entropy": self.log_message("Uyarı: Cross-entropy genellikle çıkış katmanı için 'softmax' aktivasyonu ile kullanılır.\n   Hedef (Y) verileriniz tek-sıcak (one-hot) formatında veya sınıf indeksi olarak olmalıdır."); self.update_accuracy_graph()
        except ValueError as e: 
            messagebox.showerror("Hata", str(e));
            if self.network.loss_function_name: self.loss_function_var.set(self.network.loss_function_name)

    def log_message(self, msg, clear_existing=False):
        self.log_text.config(state=tk.NORMAL)
        if clear_existing: self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, str(msg) + "\n"); self.log_text.see(tk.END); self.log_text.config(state=tk.DISABLED)

    def update_metrics_display(self, metrics_dict, final_predictions=None, final_targets=None):
        self.metrics_text.config(state=tk.NORMAL); self.metrics_text.delete(1.0, tk.END)
        if not metrics_dict and not final_predictions: self.metrics_text.insert(tk.END, "Hesaplanacak metrik yok veya eğitim yapılmadı.\n")
        else:
            for key, value in metrics_dict.items():
                if isinstance(value, float): self.metrics_text.insert(tk.END, f"{key}: {value:.4f}\n")
                else: self.metrics_text.insert(tk.END, f"{key}: {value}\n")
            if final_predictions and final_targets:
                self.metrics_text.insert(tk.END, "\nSon Örnek Tahminleri:\n")
                for i, (pred, target) in enumerate(zip(final_predictions, final_targets)):
                    pred_val_str = f"{pred:.3f}" if isinstance(pred, float) else str([f"{p:.2f}" for p in pred])
                    target_val_str = f"{target:.3f}" if isinstance(target, float) else str([f"{t:.2f}" for t in target])
                    self.metrics_text.insert(tk.END, f"  Çıktı {i+1}: Tahmin={pred_val_str}, Hedef={target_val_str}\n")
        self.metrics_text.config(state=tk.DISABLED)

    def _calculate_classification_metrics(self, all_true_one_hot, all_pred_probs, num_classes):
        if not all_true_one_hot or not all_pred_probs or num_classes == 0: return {}, None
        all_true_classes = [row.index(max(row)) for row in all_true_one_hot if row and sum(row)>0] 
        all_pred_classes = [row.index(max(row)) for row in all_pred_probs if row]

        if not all_true_classes or len(all_true_classes) != len(all_pred_classes): return {}, None
        
        tp, fp, fn = [0] * num_classes, [0] * num_classes, [0] * num_classes
        confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for true_cls, pred_cls in zip(all_true_classes, all_pred_classes):
            if 0 <= true_cls < num_classes and 0 <= pred_cls < num_classes:
                confusion_matrix[true_cls][pred_cls] += 1
                if true_cls == pred_cls: tp[true_cls] += 1
                else: fp[pred_cls] += 1; fn[true_cls] += 1
        
        precision, recall, f1 = [0.0] * num_classes, [0.0] * num_classes, [0.0] * num_classes
        for i in range(num_classes):
            precision[i] = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
            recall[i] = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0.0
        
        metrics = {
            "Precision (Macro Avg)": sum(precision) / num_classes if num_classes > 0 else 0.0,
            "Recall (Macro Avg)": sum(recall) / num_classes if num_classes > 0 else 0.0,
            "F1-score (Macro Avg)": sum(f1) / num_classes if num_classes > 0 else 0.0,
        }
        for i in range(num_classes): metrics[f"F1_Sınıf{i}"] = f1[i]
        return metrics, confusion_matrix
    
    def _display_text_confusion_matrix(self, cm, class_names=None):
        self.cm_text_area.config(state=tk.NORMAL)
        self.cm_text_area.delete(1.0, tk.END)
        if cm is None or not cm:
            self.cm_text_area.insert(tk.END, "Karmaşıklık Matrisi (Sınıflandırma eğitimi sonrası görüntülenir)")
        else:
            num_classes = len(cm)
            if class_names is None: class_names = [f"S{i}" for i in range(num_classes)]
            header = "Gerçek\\Tahmin | " + " | ".join(f"{name:^5}" for name in class_names) + "\n"
            separator = "-" * (len(header) -1) + "\n" 
            cm_str = header + separator
            for i in range(num_classes):
                row_str = f"{class_names[i]:<12} | " + " | ".join(f"{cm[i][j]:^5d}" for j in range(num_classes)) + "\n"
                cm_str += row_str
            self.cm_text_area.insert(tk.END, cm_str)
        self.cm_text_area.config(state=tk.DISABLED)


    def update_loss_graph(self):
        self.ax_loss.clear()
        if self.current_epoch_losses: self.ax_loss.plot(range(1, len(self.current_epoch_losses) + 1), self.current_epoch_losses, marker='.', linestyle='-', markersize=4, linewidth=1.5, label="Kayıp")
        self.ax_loss.set_title("Eğitim Kaybı / Epoch", fontsize=10); self.ax_loss.set_xlabel("Epoch", fontsize=9); self.ax_loss.set_ylabel("Ortalama Kayıp", fontsize=9)
        self.ax_loss.grid(True, linestyle='--', alpha=0.7); self.ax_loss.tick_params(axis='both', which='major', labelsize=8); self.ax_loss.legend(fontsize=8)
        self.fig_loss.tight_layout(); self.loss_canvas_widget.draw()
        
    def update_accuracy_graph(self):
        self.ax_accuracy.clear()
        if self.current_epoch_accuracies: self.ax_accuracy.plot(range(1, len(self.current_epoch_accuracies) + 1), self.current_epoch_accuracies, marker='.', linestyle='-', color='green', markersize=4, linewidth=1.5, label="Doğruluk")
        self.ax_accuracy.set_title("Eğitim Doğruluğu / Epoch", fontsize=10); self.ax_accuracy.set_xlabel("Epoch", fontsize=9); self.ax_accuracy.set_ylabel("Doğruluk", fontsize=9)
        self.ax_accuracy.set_ylim(0, 1.05); self.ax_accuracy.grid(True, linestyle='--', alpha=0.7); self.ax_accuracy.tick_params(axis='both', which='major', labelsize=8); self.ax_accuracy.legend(fontsize=8)
        self.fig_accuracy.tight_layout(); self.accuracy_canvas_widget.draw()

    def update_layer_config_entries(self):
        for wt in self.layer_entries: [w.destroy() for w in wt[:4]]
        self.layer_entries.clear(); num_hidden = self.num_hidden_layers_var.get()
        if num_hidden < 0 : num_hidden = 0 
        for i in range(num_hidden):
            nl = ttk.Label(self.layer_config_frame, text=f"Gizli K.{i+1} Nöron:"); nl.grid(row=i, column=0, sticky=tk.W, padx=2, pady=1)
            nv = tk.IntVar(value=3); ne = ttk.Entry(self.layer_config_frame, textvariable=nv, width=5); ne.grid(row=i, column=1, sticky=tk.EW, padx=2, pady=1)
            al = ttk.Label(self.layer_config_frame, text="Aktv:"); al.grid(row=i, column=2, sticky=tk.W, padx=2, pady=1)
            av = tk.StringVar(value="relu"); ac = ttk.Combobox(self.layer_config_frame, textvariable=av, values=list(ACTIVATION_FUNCTIONS.keys()), state="readonly", width=8); ac.grid(row=i, column=3, sticky=tk.EW, padx=2, pady=1)
            self.layer_entries.append((nl, ne, al, ac, nv, av))
        self.master.update_idletasks()
        if hasattr(self, 'left_canvas'): self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))


    def build_and_draw_network(self, custom_weights=None, custom_biases=None, custom_layer_configs_full=None, training_state=None):
        try:
            input_size = self.input_size_var.get(); layer_configs_for_nn = [] 
            if custom_layer_configs_full: 
                output_size, output_act = custom_layer_configs_full[-1][0], custom_layer_configs_full[-1][1]
                self.output_size_var.set(output_size); self.output_activation_var.set(output_act)
                hidden_cfgs_gui = custom_layer_configs_full[:-1] 
                self.num_hidden_layers_var.set(len(hidden_cfgs_gui)); self.update_layer_config_entries() 
                for i, (n, act) in enumerate(hidden_cfgs_gui):
                    if i < len(self.layer_entries): self.layer_entries[i][4].set(n); self.layer_entries[i][5].set(act) 
                layer_configs_for_nn = custom_layer_configs_full
            else: 
                output_size, output_act = self.output_size_var.get(), self.output_activation_var.get()
                for entry in self.layer_entries: 
                    num_n, act = entry[4].get(), entry[5].get();
                    if num_n <= 0: raise ValueError("Gizli katman nöron sayısı pozitif olmalı.")
                    layer_configs_for_nn.append((num_n, act))
                layer_configs_for_nn.append((output_size, output_act)) 
            if input_size <=0 or layer_configs_for_nn[-1][0] <= 0: raise ValueError("Giriş ve çıkış nöron sayıları pozitif olmalı.")
            self.network.set_loss_function(self.loss_function_var.get())
            self.network.configure_network(input_size, layer_configs_for_nn, custom_weights, custom_biases)
            self.log_message("Ağ yapısı oluşturuldu/yüklendi.", True)
            if not custom_weights: self.log_message("Ağırlıklar/biaslar rastgele/varsayılan yöntemle atandı.")
            if training_state:
                self.current_epoch_losses = training_state.get("epoch_losses", [])
                self.current_epoch_accuracies = training_state.get("epoch_accuracies", [])
                self.epochs_var.set(training_state.get("total_epochs_completed", self.epochs_var.get()))
                self.network.adam_t = training_state.get("optimizer_adam_t", 0)
                for attr_name in ["velocity_W", "velocity_b", "m_W", "v_W", "m_b", "v_b"]:
                    saved_val = training_state.get(f"optimizer_{attr_name}")
                    if saved_val is not None and hasattr(self.network, attr_name) and len(saved_val) == len(getattr(self.network, attr_name)):
                        setattr(self.network, attr_name, saved_val)
                self.log_message("Kaydedilmiş eğitim durumu yüklendi.")

            self._populate_wb_combo(); self.draw_network_on_canvas(); self.save_canvas_button.config(state=tk.NORMAL)
            for btn_name in ["forward_step_button", "forward_all_button", "train_button", "train_step_by_step_button", "save_network_button"]: getattr(self,btn_name).config(state=tk.NORMAL)
            for btn_name in ["backward_step_button", "train_next_step_button"]: getattr(self,btn_name).config(state=tk.DISABLED)
            self.current_training_phase_label.config(text="Aşama: -"); self.forward_pass_gen, self.backward_pass_gen = None, None; self.is_training_step_by_step_active = False
            if not training_state: self.current_epoch_losses, self.current_epoch_accuracies = [], []
            self.update_loss_graph(); self.update_accuracy_graph(); self._display_text_confusion_matrix(None); self.update_metrics_display({})
        except ValueError as e: messagebox.showerror("Giriş Hatası", str(e))
        except Exception as e: messagebox.showerror("Hata", f"Ağ oluşturulurken/yüklenirken: {e}"); import traceback; traceback.print_exc()

    def draw_network_on_canvas(self):
        self.canvas.delete("all")
        for store in [self.neuron_canvas_objects,self.neuron_value_texts,self.neuron_z_value_texts,self.connection_canvas_objects,self.connection_weight_value_texts,self.neuron_bias_value_texts]: store.clear()
        if not self.network or not self.network.layer_configs:
            cw,ch = self.canvas.winfo_width(),self.canvas.winfo_height(); cw,ch = (800,600) if cw<=1 or ch<=1 else (cw,ch)
            self.canvas.create_text(cw/2,ch/2,text="Ağ kurulmadı.",font=('Helvetica',16),fill="grey"); self.canvas.config(scrollregion=(0,0,cw,ch)); return
        cw,ch,nr,nl = max(800,self.canvas.winfo_width()),max(600,self.canvas.winfo_height()),20,len(self.network.layer_configs)+1
        ls = (cw-120)/max(1,nl-1) if nl>1 else cw/2; yp=80; all_nc=[self.input_size_var.get()]+[lc[0] for lc in self.network.layer_configs]; max_n=max(all_nc) if all_nc else 1
        nvs = max(3*nr,(ch-2*yp)/max(1,max_n-1 if max_n>1 else 1 )); n_pos_by_l=[]
        lx,ni = 60,self.input_size_var.get(); curr_l_pos=[]; th_in=(ni-1)*nvs; lys_in=(ch-th_in)/2
        if ni==1: lys_in=ch/2
        for i in range(ni):
            y=lys_in+i*nvs; k,name = ("input",i),self._get_neuron_display_name("Giriş",i)
            oid=self.canvas.create_oval(lx-nr,y-nr,lx+nr,y+nr,fill="lightblue",outline="black",width=1.5,tags=("neuron",f"input_{i}"))
            self.canvas.create_text(lx,y-nr-12,text=name,font=('Helvetica',8,'bold'),tags=("neuron_label",f"input_{i}_label"))
            curr_l_pos.append((lx,y)); self.neuron_canvas_objects[k]=oid
            if self.show_neuron_values_on_canvas_var.get(): self.neuron_value_texts[k]=self.canvas.create_text(lx,y+nr+10,text="",font=('Helvetica',7),tags=("neuron_value",f"input_{i}_value_a"))
        n_pos_by_l.append(curr_l_pos)
        for l_cfg_idx,(num_n,act_s) in enumerate(self.network.layer_configs):
            lx+=ls; prev_l_pos=n_pos_by_l[-1]; curr_l_pos=[]; th_l=(num_n-1)*nvs; lys_l=(ch-th_l)/2
            if num_n==1: lys_l=ch/2
            l_disp_name,l_type=self._get_layer_display_name(l_cfg_idx),"Çıkış" if "Çıkış" in self._get_layer_display_name(l_cfg_idx) else "Gizli"
            l_num_h=int(l_disp_name.split(" ")[-1]) if "Gizli" in l_disp_name else None
            for n_idx in range(num_n):
                y=lys_l+n_idx*nvs; k,name=(l_cfg_idx,n_idx),self._get_neuron_display_name(l_type,n_idx,l_num_h)
                fill_c="lightcoral" if (l_type=="Çıkış") else "lightgreen"
                oid=self.canvas.create_oval(lx-nr,y-nr,lx+nr,y+nr,fill=fill_c,outline="black",width=1.5,tags=("neuron",f"layer{l_cfg_idx}_neuron{n_idx}"))
                self.canvas.create_text(lx,y-nr-24,text=name,font=('Helvetica',7,'bold')); self.canvas.create_text(lx,y-nr-12,text=f"Akt:{act_s[:5]}.",font=('Helvetica',7,'italic'))
                curr_l_pos.append((lx,y)); self.neuron_canvas_objects[k]=oid
                if self.show_biases_on_canvas_var.get(): self.neuron_bias_value_texts[k]=self.canvas.create_text(lx,y+nr+26,text=f"b={self.network.biases[l_cfg_idx][n_idx]:.2f}",font=('Helvetica',7),fill="teal",tags=("bias_text"))
                if self.show_neuron_values_on_canvas_var.get():
                    self.neuron_value_texts[k]=self.canvas.create_text(lx,y+nr+8,text="",font=('Helvetica',7),tags=("neuron_value"))
                    self.neuron_z_value_texts[k]=self.canvas.create_text(lx,y+nr+17,text="",font=('Helvetica',7),fill="darkblue",tags=("neuron_value_z"))
                for prev_n_idx,(px,py) in enumerate(prev_l_pos):
                    w=self.network.weights[l_cfg_idx][prev_n_idx][n_idx]
                    lw,lc=min(5,max(0.5,1+abs(w)*1.5)),"darkred" if w<0 else ("darkgreen" if w>0 else "grey")
                    conn_id=self.canvas.create_line(px+nr,py,lx-nr,y,fill=lc,width=lw,arrow=tk.LAST,arrowshape=(8,10,3),tags=("connection"))
                    self.canvas.tag_lower(conn_id); self.connection_canvas_objects[(l_cfg_idx,n_idx,prev_n_idx)]=(conn_id,lc,lw) 
                    if self.show_weights_on_canvas_var.get():
                        mid_x,mid_y=(px+nr+lx-nr)/2,(py+y)/2; ang=math.atan2(y-py,lx-nr-(px+nr))*180/math.pi
                        dyo,dxo=-8,8
                        if -20<ang<20 or ang>160 or ang<-160: mid_y+=dyo
                        else: mid_x+=dxo*(-1 if (0<ang<160) else 1)
                        self.connection_weight_value_texts[(l_cfg_idx,n_idx,prev_n_idx)]=self.canvas.create_text(mid_x,mid_y,text=f"{w:.2f}",fill="purple",font=('Helvetica',7,'bold'),angle=ang,tags="weight_text")
            n_pos_by_l.append(curr_l_pos)
        bbox=self.canvas.bbox("all")
        if bbox: self.canvas.config(scrollregion=(bbox[0]-30,bbox[1]-30,bbox[2]+30,bbox[3]+50)) 
        else: self.canvas.config(scrollregion=(0,0,cw,ch))
        self.master.update_idletasks()

    def _parse_input_data(self, text_data_str, num_features, is_target=False, num_output_for_one_hot=0):
        samples=[]
        for line_idx,line_raw in enumerate(text_data_str.strip().split(';')):
            line=line_raw.strip()
            if not line: continue
            try:
                vals_f=[float(v.strip()) for v in line.split(',')]
                if is_target and self.loss_function_var.get()=="cross_entropy":
                    if num_output_for_one_hot<=0: raise ValueError("CE için çıkış sınıf sayısı > 0 olmalı.")
                    if len(vals_f)==1 and 0<=vals_f[0]<num_output_for_one_hot and vals_f[0]==int(vals_f[0]):
                        oh=[0.0]*num_output_for_one_hot; oh[int(vals_f[0])]=1.0; samples.append(oh)
                    elif len(vals_f)==num_output_for_one_hot: samples.append(vals_f)
                    else: raise ValueError(f"Satır {line_idx+1} (Hedef): CE için {num_output_for_one_hot} elemanlı one-hot veya tek sınıf indeksi beklenir.")
                else: 
                    if len(vals_f)!=num_features: raise ValueError(f"Satır {line_idx+1}: Beklenen {num_features} özellik/hedef, bulunan {len(vals_f)}.")
                    samples.append(vals_f)
            except ValueError as e: raise ValueError(f"Veri ayrıştırma hatası (Satır {line_idx+1}: '{line_raw}'): {e}")
        return samples

    def _get_first_training_sample_for_step_ops(self): 
        x_str,y_str=self.x_input_text.get(1.0,tk.END).strip().split(';')[0],self.y_input_text.get(1.0,tk.END).strip().split(';')[0]
        if not x_str: messagebox.showinfo("Bilgi","Lütfen geçerli bir giriş verisi (X) girin."); return None,None
        try:
            x_s=self._parse_input_data(x_str,self.input_size_var.get())[0]
            y_s=self._parse_input_data(y_str,self.output_size_var.get(),True,self.output_size_var.get())[0] if y_str else None
            return x_s,y_s
        except Exception as e: messagebox.showerror("Veri Okuma Hatası",f"İlk örnek okunurken: {e}"); return None,None

    def execute_forward_step(self):
        try:
            if self.forward_pass_gen is None: 
                x,_=self._get_first_training_sample_for_step_ops()
                if x is None: return
                self.log_message(f"\nİleri Yayılım Adımı Başlatılıyor. Giriş: {[f'{v:.3f}' for v in x]}",True)
                self.current_training_phase_label.config(text="Aşama: İleri (Adım)")
                self.network.current_input_for_forward=x; self.forward_pass_gen=self.network.forward_pass_generator(x,self.detailed_forward_steps.get())
                self.reset_neuron_visuals_and_texts(True); self.highlight_step_on_canvas(None) 
                self.backward_step_button.config(state=tk.DISABLED); self.backward_pass_gen=None
            res=next(self.forward_pass_gen,None)
            self.handle_forward_step_result_and_visualize(res,phase_override="İleri Yayılım")
        except Exception as e: messagebox.showerror("Hata",f"İleri adım: {e}"); import traceback; traceback.print_exc(); self.forward_pass_gen=None; self.reset_neuron_visuals_and_texts(); self.highlight_step_on_canvas(None); self.current_training_phase_label.config(text="Aşama: -")

    def execute_forward_all(self):
        try:
            x,_=self._get_first_training_sample_for_step_ops()
            if x is None: return
            self.log_message(f"\nTüm İleri Yayılım. Giriş: {[f'{v:.3f}' for v in x]}",True)
            self.current_training_phase_label.config(text="Aşama: İleri (Tümü)")
            self.reset_neuron_visuals_and_texts(True); self.highlight_step_on_canvas(None)
            self.network.current_input_for_forward=x; gen=self.network.forward_pass_generator(x,False); final_out=None
            for res in gen: 
                self.handle_forward_step_result_and_visualize(res,True,"İleri");
                if res["type"]=="forward_pass_complete": final_out=res["final_output"]
            self.log_message(f"İleri yayılım tamamlandı. Sonuç: {[f'{o:.3f}' for o in final_out]}" if final_out else "İleri yayılım sonucu yok.")
            self.forward_pass_gen=None; self.reset_neuron_visuals_and_texts(False); self.highlight_step_on_canvas(None) 
            _,y=self._get_first_training_sample_for_step_ops()
            if y and self.network.neuron_outputs_a: self.backward_step_button.config(state=tk.NORMAL)
            self.backward_pass_gen=None; self.current_training_phase_label.config(text="Aşama: - (İleri Bitti)")
        except Exception as e: messagebox.showerror("Hata",f"İleri yayılım: {e}"); import traceback; traceback.print_exc(); self.reset_neuron_visuals_and_texts(); self.current_training_phase_label.config(text="Aşama: -")

    def handle_forward_step_result_and_visualize(self, step_res, log_min=False, phase_override=None):
        if phase_override: self.current_training_phase_label.config(text=f"Aşama: {phase_override}")
        if not step_res: 
            self.log_message("İleri yayılım adımları tamamlandı."); self.forward_pass_gen=None
            self.reset_neuron_visuals_and_texts(False); self.highlight_step_on_canvas(None)
            _,y=self._get_first_training_sample_for_step_ops()
            if y and self.network.neuron_outputs_a: self.backward_step_button.config(state=tk.NORMAL)
            self.current_training_phase_label.config(text="Aşama: - (İleri Bitti)"); return
        log_pref="[İleri] "; 
        if not log_min: self.log_message(f"{log_pref}Adım: {step_res['type']}",clear_existing=log_min and step_res['type']!='input_layer')
        self.highlight_step_on_canvas(step_res) 
        if step_res["type"]=="input_layer":
            if not log_min: self.log_message(f"{log_pref} Giriş (a): {[f'{x:.3f}' for x in step_res['outputs']]}")
            if self.show_neuron_values_on_canvas_var.get(): [self.canvas.itemconfig(self.neuron_value_texts[("input",i)],text=f"a={val:.2f}") for i,val in enumerate(step_res['outputs']) if ("input",i) in self.neuron_value_texts]
        elif step_res["type"]=="weight_multiplication":
            l,n,prev_n=step_res["layer_index"],step_res["neuron_index"],step_res["prev_neuron_index"]
            t_name,s_name=self._get_layer_display_name(l),self._get_source_layer_display_name_for_weights(l)
            t_neuron=self._get_neuron_display_name("Çıkış" if t_name=="Çıkış Katmanı" else "Gizli",n,l+1 if "Gizli" in t_name else None)
            s_neuron=self._get_neuron_display_name("Giriş" if s_name=="Giriş Katmanı" else "Gizli",prev_n,l if "Gizli" in s_name else None)
            if not log_min: self.log_message(f"{log_pref} {t_neuron} <- {s_neuron}: a_s={step_res['prev_activation']:.2f}*w={step_res['weight']:.2f}={step_res['product']:.2f}. ΣZ={step_res['current_sum_for_neuron_z']:.2f}")
        elif step_res["type"]=="bias_addition":
            l,n=step_res["layer_index"],step_res["neuron_index"]; t_name=self._get_layer_display_name(l)
            t_neuron=self._get_neuron_display_name("Çıkış" if t_name=="Çıkış Katmanı" else "Gizli",n,l+1 if "Gizli" in t_name else None)
            if not log_min: self.log_message(f"{log_pref} {t_neuron}: z_nob={step_res['z_unbiased']:.2f}+b={step_res['bias']:.2f}=z={step_res['z_final']:.2f}")
            if self.show_neuron_values_on_canvas_var.get() and (l,n) in self.neuron_z_value_texts: self.canvas.itemconfig(self.neuron_z_value_texts[(l,n)],text=f"z={step_res['z_final']:.2f}")
        elif step_res["type"]=="layer_activation":
            l,name=step_res["layer_index"],self._get_layer_display_name(step_res["layer_index"])
            if not log_min: self.log_message(f"{log_pref} {name} ({step_res['activation_function']}) Hesaplandı:\n{log_pref}   z: {[f'{x:.3f}' for x in step_res['z_values']]}\n{log_pref}   a: {[f'{x:.3f}' for x in step_res['a_values']]}")
            if self.show_neuron_values_on_canvas_var.get():
                for n,(z,a) in enumerate(zip(step_res["z_values"],step_res["a_values"])):
                    if (l,n) in self.neuron_z_value_texts: self.canvas.itemconfig(self.neuron_z_value_texts[(l,n)],text=f"z={z:.2f}")
                    if (l,n) in self.neuron_value_texts: self.canvas.itemconfig(self.neuron_value_texts[(l,n)],text=f"a={a:.2f}")
        elif step_res["type"]=="forward_pass_complete" and not log_min: self.log_message(f"{log_pref}İleri yayılım bitti. Sonuç: {[f'{x:.3f}' for x in step_res['final_output']]}")

    def execute_backward_step(self):
        try:
            if self.backward_pass_gen is None:
                _,y=self._get_first_training_sample_for_step_ops()
                if y is None: messagebox.showinfo("Bilgi","Geri yayılım için hedef (Y) verisi gerekli."); return
                if not self.network.neuron_outputs_a or len(self.network.neuron_outputs_a)<=1: messagebox.showinfo("Bilgi","Geri yayılım için önce ileri yayılım çalıştırılmalı."); return
                self.log_message(f"\nGeri Yayılım Adımı Başlatılıyor. Hedef: {[f'{v:.3f}' for v in y]}",True)
                self.current_training_phase_label.config(text="Aşama: Geri (Adım)")
                lr,opt=self.lr_var.get(),{"type":self.optimizer_var.get(),"beta":0.9,"beta1":0.9,"beta2":0.999,"epsilon":1e-8} 
                self.backward_pass_gen=self.network.backward_pass_generator(y,lr,opt)
                self.reset_neuron_visuals_and_texts(False); self.highlight_step_on_canvas(None)
            res=next(self.backward_pass_gen,None)
            self.handle_backward_step_result_and_visualize(res,current_phase_override="Geri Yayılım")
        except Exception as e: messagebox.showerror("Hata",f"Geri adım: {e}"); import traceback; traceback.print_exc(); self.backward_pass_gen=None; self.reset_neuron_visuals_and_texts(False); self.highlight_step_on_canvas(None); self.current_training_phase_label.config(text="Aşama: -")

    def handle_backward_step_result_and_visualize(self, step_result, current_phase_override=None):
        if current_phase_override: self.current_training_phase_label.config(text=f"Aşama: {current_phase_override}")
        log_msg_prefix = "[Geri Yayılım] "
        if not step_result: 
            self.log_message(f"{log_msg_prefix}Geri yayılım adımları tamamlandı."); self.backward_pass_gen = None
            self.reset_neuron_visuals_and_texts(False); self.highlight_step_on_canvas(None)
            self.draw_network_on_canvas(); self._populate_wb_combo() 
            self.current_training_phase_label.config(text="Aşama: - (Geri Bitti)"); return
        
        self.log_message(f"{log_msg_prefix}Adım: {step_result['type']}"); self.highlight_step_on_canvas(step_result); l_cfg = step_result.get("layer_index", -1) 
        
        if step_result["type"] == "output_delta_calculation" or step_result["type"] == "hidden_delta_calculation":
            m, name = step_result.get('method', ''), self._get_layer_display_name(l_cfg)
            self.log_message(f"{log_msg_prefix} {name} Delta (δ) Hesaplanıyor ({m}):")
            for k in ['dL_daL', 'f_prime_z_L', 'f_prime_z_l', 'error_propagated']: 
                if k in step_result: self.log_message(f"{log_msg_prefix}   {k}: {[f'{x:.3f}' for x in step_result[k]]}")
            delta_k = 'delta_L' if 'delta_L' in step_result else 'delta_l'
            self.log_message(f"{log_msg_prefix}   δ: {[f'{x:.3f}' for x in step_result[delta_k]]}")
            if self.show_neuron_values_on_canvas_var.get():
                for n, dv in enumerate(step_result[delta_k]):
                    if (l_cfg, n) in self.neuron_z_value_texts: self.canvas.itemconfig(self.neuron_z_value_texts[(l_cfg, n)], text=f"δ={dv:.2f}", fill="purple")
        elif step_result["type"] == "gradient_calculation": 
            self.log_message(f"{log_msg_prefix} {self._get_layer_display_name(l_cfg)} Gradyanları (dW, db): dW({step_result['grad_W_l_dims']}), db({step_result['grad_b_l_dims']})")
        elif step_result["type"] == "weight_update": 
            self.log_message(f"{log_msg_prefix} {self._get_layer_display_name(l_cfg)} Ağırlık/Bias Güncellendi (Opt: {step_result['optimizer_used']}).")

    def start_step_by_step_training_one_sample(self):
        self.is_training_step_by_step_active=True; self.forward_pass_gen,self.backward_pass_gen=None,None 
        x,y=self._get_first_training_sample_for_step_ops()
        if x is None or y is None: self.is_training_step_by_step_active=False; messagebox.showinfo("Bilgi","Adım adım eğitim için X ve Y verisi gerekli."); self.current_training_phase_label.config(text="Aşama: -"); return
        self.current_training_X_sample,self.current_training_Y_sample=x,y
        self.log_message(f"\nAdım Adım Eğitim (1 Örnek) Başlatılıyor.",True); self.log_message(f"  Giriş (X): {[f'{v:.3f}' for v in x]}"); self.log_message(f"  Hedef (Y): {[f'{v:.3f}' for v in y]}"); self.log_message("  İlk adım: İleri Yayılım. 'Eğitimde Sonraki Adım >>' butonuna basın.")
        self.current_training_phase_label.config(text="Aşama: İleri (Bekliyor)")
        for btn in [self.train_next_step_button]: btn.config(state=tk.NORMAL)
        for btn in [self.train_step_by_step_button,self.train_button,self.forward_step_button,self.backward_step_button,self.forward_all_button]: btn.config(state=tk.DISABLED)
        self.network.current_input_for_forward=self.current_training_X_sample
        self.forward_pass_gen=self.network.forward_pass_generator(self.current_training_X_sample,self.detailed_forward_steps.get())
        self.reset_neuron_visuals_and_texts(True); self.highlight_step_on_canvas(None)

    def execute_next_training_step(self):
        if not self.is_training_step_by_step_active: return
        try:
            if self.forward_pass_gen: 
                res=next(self.forward_pass_gen,None); self.handle_forward_step_result_and_visualize(res,phase_override="İleri (Adım)") 
                if res and res["type"]=="forward_pass_complete":
                    self.forward_pass_gen=None; self.log_message("  İleri yayılım tamamlandı. Sonraki adım: Geri Yayılım.")
                    self.current_training_phase_label.config(text="Aşama: Geri (Bekliyor)")
                    lr,opt=self.lr_var.get(),{"type":self.optimizer_var.get(),"beta":0.9,"beta1":0.9,"beta2":0.999,"epsilon":1e-8}
                    self.backward_pass_gen=self.network.backward_pass_generator(self.current_training_Y_sample,lr,opt)
                elif not res: self.forward_pass_gen=None 
            elif self.backward_pass_gen: 
                res=next(self.backward_pass_gen,None); self.handle_backward_step_result_and_visualize(res,current_phase_override="Geri (Adım)")
                if res and res["type"]=="backward_pass_complete":
                    self.backward_pass_gen=None; self.log_message("  Geri yayılım tamamlandı. Bu örneğin eğitimi bitti."); self.log_message("  Yeni bir örnekle eğitime başlamak için 'Adım Adım Eğit'i kullanın.")
                    self.current_training_phase_label.config(text="Aşama: - (Örnek Bitti)"); self.is_training_step_by_step_active=False
                    for btn in [self.train_step_by_step_button,self.train_button,self.forward_step_button,self.forward_all_button]: btn.config(state=tk.NORMAL)
                    self.train_next_step_button.config(state=tk.DISABLED)
                elif not res: self.backward_pass_gen=None 
            else: 
                self.log_message("Adım adım eğitim için 'Adım Adım Eğit (1 Örnek Başlat)' butonuna basın."); self.current_training_phase_label.config(text="Aşama: -"); self.is_training_step_by_step_active=False
                for btn in [self.train_step_by_step_button,self.train_button,self.forward_step_button,self.forward_all_button]: btn.config(state=tk.NORMAL)
                self.train_next_step_button.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Hata",f"Eğitim adımı: {e}"); import traceback; traceback.print_exc(); self.is_training_step_by_step_active=False; self.forward_pass_gen,self.backward_pass_gen=None,None
            for btn in [self.train_step_by_step_button,self.train_button,self.forward_step_button,self.forward_all_button]: btn.config(state=tk.NORMAL)
            self.train_next_step_button.config(state=tk.DISABLED); self.reset_neuron_visuals_and_texts(); self.highlight_step_on_canvas(None); self.current_training_phase_label.config(text="Aşama: Hata")

    def start_training_auto(self):
        orig_btn_states={} 
        try:
            n_epochs,lr=self.epochs_var.get(),self.lr_var.get(); in_f,out_f=self.input_size_var.get(),self.output_size_var.get()
            if not self.training_data_X or not self.training_data_Y: 
                X_str,Y_str=self.x_input_text.get(1.0,tk.END),self.y_input_text.get(1.0,tk.END)
                X_train,Y_train=self._parse_input_data(X_str,in_f),self._parse_input_data(Y_str,out_f,True,out_f)
            else: X_train,Y_train=self.training_data_X,self.training_data_Y
            if not X_train or not Y_train: raise ValueError("Eğitim için X ve Y verileri sağlanmalıdır.")
            if len(X_train)!=len(Y_train): raise ValueError("X ve Y veri örnek sayıları eşleşmelidir.")
            self.log_message(f"\nOtomatik Eğitim başlatılıyor... Epoch: {n_epochs}, LR: {lr}",True); self.current_training_phase_label.config(text="Aşama: Otomatik Eğitim")
            self.current_epoch_losses,self.current_epoch_accuracies=[],[]; self.update_loss_graph(); self.update_accuracy_graph(); self.progress_bar["value"]=0; self.progress_bar["maximum"]=n_epochs
            opt_params={"type":self.optimizer_var.get(),"beta":0.9,"beta1":0.9,"beta2":0.999,"epsilon":1e-8}
            btns_disable=[self.build_network_button,self.load_network_button,self.load_csv_button,self.forward_step_button,self.forward_all_button,self.backward_step_button,self.train_step_by_step_button,self.train_next_step_button,self.train_button,self.wb_apply_button]
            for btn in btns_disable:
                if hasattr(self,btn.winfo_name()): orig_btn_states[btn]=btn.cget('state'); btn.config(state=tk.DISABLED)
            self.master.update()
            watch,delay=self.auto_train_watch_steps_var.get(),self.auto_train_step_delay_var.get() if self.auto_train_watch_steps_var.get() else 0
            all_true_for_cm, all_pred_for_cm = [], []

            for epoch in range(n_epochs):
                self.progress_bar["value"]=epoch+1
                loss_sum,n_correct,epoch_true_cm,epoch_pred_cm=0.0,0,[],[]
                data=list(zip(X_train,Y_train)); random.shuffle(data); X_shuff,Y_shuff=zip(*data)
                for i in range(len(X_shuff)):
                    x,y=X_shuff[i],Y_shuff[i]
                    if watch:
                        self.log_message(f"[E{epoch+1},Ö{i+1}] İleri...",True if i==0 else False); self.current_training_phase_label.config(text=f"Oto:E{epoch+1} Ö{i+1} İleri")
                        fwd_gen=self.network.forward_pass_generator(x,self.detailed_forward_steps.get())
                        for res in fwd_gen: 
                            self.handle_forward_step_result_and_visualize(res,not self.detailed_forward_steps.get(),f"Oto.E{epoch+1} Ö{i+1} İleri Adım");
                            if delay>0: time.sleep(delay); self.master.update()
                    else: list(self.network.forward_pass_generator(x,False))
                    preds=self.network.neuron_outputs_a[-1]; loss_sum+=self.network.loss_func(y,preds)
                    if self.loss_function_var.get()=="cross_entropy" and self.network.layer_configs[-1][1]=="softmax":
                        if preds and y and sum(y)>0: 
                            pred_cls,true_cls=preds.index(max(preds)),y.index(max(y)) 
                            epoch_true_cm.append(true_cls); epoch_pred_cm.append(pred_cls)
                            if pred_cls==true_cls: n_correct+=1
                    if watch:
                        self.log_message(f"[E{epoch+1},Ö{i+1}] Geri..."); self.current_training_phase_label.config(text=f"Oto:E{epoch+1} Ö{i+1} Geri")
                        bwd_gen=self.network.backward_pass_generator(y,lr,opt_params)
                        for res in bwd_gen: 
                            self.handle_backward_step_result_and_visualize(res,current_phase_override=f"Oto.E{epoch+1} Ö{i+1} Geri Adım");
                            if delay>0: time.sleep(delay); self.master.update()
                        if i%(len(X_shuff)//5+1)==0: self.draw_network_on_canvas()
                    else: list(self.network.backward_pass_generator(y,lr,opt_params))
                
                if epoch == n_epochs -1 : 
                    all_true_for_cm.extend(epoch_true_cm)
                    all_pred_for_cm.extend(epoch_pred_cm)

                avg_loss=loss_sum/len(X_shuff); self.current_epoch_losses.append(avg_loss); metrics={"Ort. Kayıp":avg_loss}
                if self.loss_function_var.get()=="cross_entropy" and self.network.layer_configs[-1][1]=="softmax":
                    acc=n_correct/len(X_shuff) if X_shuff else 0; self.current_epoch_accuracies.append(acc); metrics["Doğruluk"]=acc
                log_int=max(1,n_epochs//20 if n_epochs>=20 else 1) 
                if (epoch+1)%log_int==0 or epoch==n_epochs-1: 
                    log_s=f"Epoch {epoch+1}/{n_epochs}, Ort.Kayıp: {avg_loss:.6f}"; 
                    if "Doğruluk" in metrics: log_s+=f", Doğruluk: {metrics['Doğruluk']:.4f}"
                    self.log_message(log_s); self.update_metrics_display(metrics); self.update_loss_graph(); self.update_accuracy_graph()
                    if not watch: self.master.update_idletasks()
            self.log_message("Eğitim tamamlandı.")
            final_preds, final_targets = None, None
            if X_train: final_preds = self.network.neuron_outputs_a[-1] if self.network.neuron_outputs_a else []; final_targets = Y_train[0]
            
            final_metrics_for_display = {"Ort. Kayıp (Son Epoch)": self.current_epoch_losses[-1]} if self.current_epoch_losses else {}
            if self.current_epoch_accuracies: final_metrics_for_display["Doğruluk (Son Epoch)"] = self.current_epoch_accuracies[-1]

            if self.loss_function_var.get() == "cross_entropy" and all_true_for_cm:
                num_classes_cm = self.output_size_var.get()
                cls_metrics, conf_matrix = self._calculate_classification_metrics(all_true_for_cm, all_pred_for_cm, num_classes_cm)
                final_metrics_for_display.update(cls_metrics)
                self._display_text_confusion_matrix(conf_matrix, [f"S{i}" for i in range(num_classes_cm)])
            self.update_metrics_display(final_metrics_for_display, final_preds, final_targets)


            self.current_training_phase_label.config(text="Aşama: - (Oto. Eğitim Bitti)"); self._populate_wb_combo(); self.draw_network_on_canvas() 
            if X_train: 
                self.x_input_text.delete(1.0,tk.END); self.x_input_text.insert(tk.END,",".join(map(str,X_train[0])))
                self.y_input_text.delete(1.0,tk.END); y_d=Y_train[0]
                y_s=str(y_d.index(1.0)) if self.loss_function_var.get()=="cross_entropy" and isinstance(y_d,list) and 1.0 in y_d else (",".join(map(str,y_d)) if isinstance(y_d,list) else str(y_d))
                self.y_input_text.insert(tk.END,y_s); self.execute_forward_all() 
        except ValueError as e: messagebox.showerror("Giriş Hatası",f"Veri/Eğitim: {str(e)}")
        except Exception as e: messagebox.showerror("Hata",f"Eğitim: {e}"); import traceback; traceback.print_exc()
        finally: 
            for btn,state in orig_btn_states.items():
                if btn.winfo_exists(): btn.config(state=state)
            self.master.update()

    def highlight_step_on_canvas(self, step_res):
        self.reset_neuron_visuals_and_texts(False,True) 
        for k,(conn_id,oc,ow) in self.connection_canvas_objects.items():
            if self.canvas.winfo_exists() and conn_id in self.canvas.find_all(): self.canvas.itemconfig(conn_id,fill=oc,width=ow)
        if not step_res: return 
        st,l_cfg=step_res["type"],step_res.get("layer_index",-2); hl_key=None
        if st=="input_layer": [self.canvas.itemconfig(self.neuron_canvas_objects[("input",i)],fill="yellow") for i in range(step_res.get("num_neurons",0)) if ("input",i) in self.neuron_canvas_objects]
        elif st in ["weight_multiplication","bias_addition"]: hl_key=(l_cfg,step_res["neuron_index"])
        elif st=="layer_activation": [self.canvas.itemconfig(self.neuron_canvas_objects[(l_cfg,n)],fill="yellow") for n in range(step_res.get("num_neurons",0)) if (l_cfg,n) in self.neuron_canvas_objects]
        elif "delta_calculation" in st: [self.canvas.itemconfig(self.neuron_canvas_objects[(l_cfg,n)],fill="magenta") for n in range(step_res.get("num_neurons",0)) if (l_cfg,n) in self.neuron_canvas_objects]
        if hl_key and hl_key in self.neuron_canvas_objects: self.canvas.itemconfig(self.neuron_canvas_objects[hl_key],fill="orange")
        if st=="weight_multiplication" and self.detailed_forward_steps.get():
            prev_n_idx=step_res["prev_neuron_index"]; prev_l_key=("input" if l_cfg==0 else (l_cfg-1,prev_n_idx))
            if prev_l_key[0]=="input": 
                if ("input",prev_n_idx) in self.neuron_canvas_objects: self.canvas.itemconfig(self.neuron_canvas_objects[("input",prev_n_idx)],fill="khaki")
            elif prev_l_key in self.neuron_canvas_objects: self.canvas.itemconfig(self.neuron_canvas_objects[prev_l_key],fill="khaki")
            conn_k=(l_cfg,step_res["neuron_index"],prev_n_idx)
            if conn_k in self.connection_canvas_objects:
                conn_id,_,ow=self.connection_canvas_objects[conn_k]
                if conn_id in self.canvas.find_all(): self.canvas.itemconfig(conn_id,fill="blue",width=max(3,float(ow)+1.5))
        self.master.update_idletasks()

    def reset_neuron_visuals_and_texts(self, clear_vals=False, reset_hl_only=False, hl_layer_key=None, hl_neuron_idx=None):
        for i in range(self.input_size_var.get()):
            k=("input",i); is_hl=(hl_layer_key=="input" and hl_neuron_idx==i)
            if k in self.neuron_canvas_objects: self.canvas.itemconfig(self.neuron_canvas_objects[k],fill="yellow" if is_hl else "lightblue")
            if clear_vals and k in self.neuron_value_texts and self.show_neuron_values_on_canvas_var.get(): self.canvas.itemconfig(self.neuron_value_texts[k],text="")
        if self.network and self.network.layer_configs:
            for l_idx,(n_n,_) in enumerate(self.network.layer_configs):
                for n_i in range(n_n):
                    k=(l_idx,n_i); is_hl=(hl_layer_key==l_idx and hl_neuron_idx==n_i)
                    if k in self.neuron_canvas_objects: is_out=(l_idx==len(self.network.layer_configs)-1); def_fill="lightcoral" if is_out else "lightgreen"; self.canvas.itemconfig(self.neuron_canvas_objects[k],fill="yellow" if is_hl else def_fill)
                    if clear_vals and self.show_neuron_values_on_canvas_var.get():
                        if k in self.neuron_value_texts: self.canvas.itemconfig(self.neuron_value_texts[k],text="")
                        if k in self.neuron_z_value_texts: self.canvas.itemconfig(self.neuron_z_value_texts[k],text="",fill="darkblue")
                    elif not reset_hl_only and not is_hl and k in self.neuron_z_value_texts and self.show_neuron_values_on_canvas_var.get():
                         zid=self.neuron_z_value_texts[k]; c_txt=self.canvas.itemcget(zid,'text')
                         if "δ=" in c_txt: p=c_txt.split('\n'); z_p=p[0] if p and p[0].startswith("z=") else ""; self.canvas.itemconfig(zid,text=z_p,fill="darkblue")
        self.master.update_idletasks()

    def on_canvas_click(self, event):
        cx,cy=self.canvas.canvasx(event.x),self.canvas.canvasy(event.y)
        items=self.canvas.find_closest(cx,cy); clicked_info=None
        if items:
            item,tags,bbox=items[0],self.canvas.gettags(items[0]),self.canvas.bbox(items[0])
            if bbox and not (bbox[0]<=cx<=bbox[2] and bbox[1]<=cy<=bbox[3]): self.reset_neuron_visuals_and_texts(False,True); return
            if "neuron" in tags:
                for tag in tags:
                    if tag.startswith("input_"): 
                        try: clicked_info=("input",int(tag.split("_")[1])); break; 
                        except: pass
                    elif tag.startswith("layer"): 
                        try: p=tag.replace("layer","").split("_neuron"); clicked_info=(int(p[0]),int(p[1])); break; 
                        except: pass
        if clicked_info:
            l_key,n_idx=clicked_info[0],clicked_info[1]; disp_l_name,is_input="",(l_key=="input")
            if is_input: disp_l_name,neuron_name="Giriş Katmanı",self._get_neuron_display_name("Giriş",n_idx)
            else: l_cfg=int(l_key); disp_l_name=self._get_layer_display_name(l_cfg); neuron_name=self._get_neuron_display_name("Çıkış" if "Çıkış" in disp_l_name else "Gizli",n_idx,l_cfg+1 if "Gizli" in disp_l_name else None)
            self.log_message(f"Tıklanan Nöron: {neuron_name} ({disp_l_name})")
            self.reset_neuron_visuals_and_texts(False,True,l_key,n_idx) 
            info_str=f"Nöron: {neuron_name}\nKatman: {disp_l_name}\n"
            if is_input:
                if self.network.current_input_for_forward and n_idx < len(self.network.current_input_for_forward): info_str+=f"Değer (a): {self.network.current_input_for_forward[n_idx]:.4f}\n"
            else: 
                l_cfg=int(l_key); info_str+=f"Aktivasyon Fonk: {self.network.layer_configs[l_cfg][1]}\n"
                if self.network.neuron_outputs_z and l_cfg<len(self.network.neuron_outputs_z) and n_idx<len(self.network.neuron_outputs_z[l_cfg]): info_str+=f"Z Değeri: {self.network.neuron_outputs_z[l_cfg][n_idx]:.4f}\n"
                if self.network.neuron_outputs_a and (l_cfg+1)<len(self.network.neuron_outputs_a) and n_idx<len(self.network.neuron_outputs_a[l_cfg+1]): info_str+=f"Aktivasyon (a): {self.network.neuron_outputs_a[l_cfg+1][n_idx]:.4f}\n"
                if self.network.biases and l_cfg<len(self.network.biases) and n_idx<len(self.network.biases[l_cfg]): info_str+=f"Bias (b): {self.network.biases[l_cfg][n_idx]:.4f}\n"
                info_str+="Gelen Ağırlıklar (Kaynak → Bu Nöron):\n"
                num_prev_n,src_l_name=(self.input_size_var.get() if l_cfg==0 else self.network.layer_configs[l_cfg-1][0]),self._get_source_layer_display_name_for_weights(l_cfg)
                if self.network.weights and l_cfg<len(self.network.weights):
                    for prev_n in range(num_prev_n):
                        if prev_n<len(self.network.weights[l_cfg]) and n_idx<len(self.network.weights[l_cfg][prev_n]):
                            w=self.network.weights[l_cfg][prev_n][n_idx]; src_l_num=None
                            if "Gizli" in src_l_name: parts=src_l_name.split(" "); src_l_num=int(parts[-1]) if len(parts)>1 and parts[0]=="Gizli" else None
                            src_n_name=self._get_neuron_display_name("Giriş" if "Giriş" in src_l_name else "Gizli",prev_n,src_l_num)
                            info_str+=f"  {src_n_name} → {w:.4f}\n"
            win=tk.Toplevel(self.master); win.title("Nöron Detayları"); win.transient(self.master); win.grab_set()
            info_label = scrolledtext.ScrolledText(win, height=15, width=50, wrap=tk.WORD, font=("Calibri", 10))
            info_label.insert(tk.END, info_str)
            info_label.config(state=tk.DISABLED)
            info_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            ttk.Button(win, text="Kapat", command=win.destroy).pack(pady=5)
            win.update_idletasks(); w_win,h_win=win.winfo_reqwidth(),win.winfo_reqheight(); mw,mh=self.master.winfo_width(),self.master.winfo_height()
            xpos=self.master.winfo_x()+(mw//2)-(w_win//2); ypos=self.master.winfo_y()+(mh//2)-(h_win//2)
            win.geometry(f"{w_win}x{h_win}+{xpos}+{ypos}")
        else: self.reset_neuron_visuals_and_texts(False,True) 

    def reset_simulation(self):
        self.log_message("Simülasyon sıfırlanıyor...",True); self.network=NeuralNetwork(self.loss_function_var.get())
        self.forward_pass_gen,self.backward_pass_gen,self.is_training_step_by_step_active=None,None,False
        self.training_data_X,self.training_data_Y,self.current_training_X_sample,self.current_training_Y_sample=[],[],None,None
        self.update_layer_config_entries(); self.reset_neuron_visuals_and_texts(True); self.draw_network_on_canvas(); self._populate_wb_combo() 
        self.current_epoch_losses,self.current_epoch_accuracies=[],[]; self.update_loss_graph(); self.update_accuracy_graph(); self._display_text_confusion_matrix(None); self.update_metrics_display({})
        self.current_training_phase_label.config(text="Aşama: -"); self.progress_bar["value"]=0
        for btn_name in ["forward_step_button","forward_all_button","train_button","backward_step_button","train_step_by_step_button","train_next_step_button","save_network_button","save_canvas_button"]:
            if hasattr(self,btn_name): getattr(self,btn_name).config(state=tk.DISABLED)
        for btn_name in ["build_network_button","load_network_button","load_csv_button"]: getattr(self,btn_name).config(state=tk.NORMAL)
        self.log_message("Simülatör sıfırlandı. Yeni ağ kurun/yükleyin.\nİşlemleri buradan ve grafik sekmelerinden canlı izleyebilirsiniz.")

    def load_data_from_csv(self):
        fp=filedialog.askopenfilename(title="CSV Veri Dosyasını Seç",filetypes=(("CSV","*.csv"),("Tüm Dosyalar","*.*")),parent=self.master)
        if not fp: return
        try:
            X,Y=[],[]; num_in,num_out_user=self.input_size_var.get(),self.output_size_var.get() 
            with open(fp,'r',newline='',encoding='utf-8-sig') as f: 
                r,h=csv.reader(f),next(csv.reader(f),None) 
                if h: self.log_message(f"CSV başlığı: {h}")
                for idx,row in enumerate(r):
                    if not row or len(row)<num_in+1: self.log_message(f"Uyarı: Satır {idx+1} yetersiz/boş, atlanıyor."); continue
                    try:
                        x_v=[float(v.strip()) for v in row[:num_in]]; y_raw_f=[float(v.strip()) for v in row[num_in:]]; y_parsed=[]
                        if self.loss_function_var.get()=="cross_entropy":
                            if num_out_user<=0: raise ValueError("CE için çıkış sınıf sayısı > 0 olmalı.")
                            if len(y_raw_f)==1 and 0<=y_raw_f[0]<num_out_user and y_raw_f[0]==int(y_raw_f[0]): oh=[0.0]*num_out_user; oh[int(y_raw_f[0])]=1.0; y_parsed=oh
                            elif len(y_raw_f)==num_out_user: y_parsed=y_raw_f
                            else: raise ValueError(f"Satır {idx+1} (Hedef): CE için {num_out_user} elemanlı one-hot veya tek sınıf indeksi beklenir.")
                        else: 
                            if len(y_raw_f)!=num_out_user: raise ValueError(f"MSE vb. için Y formatı ({len(y_raw_f)}) çıkış nöron sayısıyla ({num_out_user}) eşleşmiyor.")
                            y_parsed=y_raw_f
                        X.append(x_v); Y.append(y_parsed)
                    except ValueError as ve: self.log_message(f"Uyarı: Satır {idx+1} hatalı değer içeriyor, atlanıyor: {ve}"); continue
            self.training_data_X,self.training_data_Y=X,Y; self.x_input_text.delete(1.0,tk.END); self.y_input_text.delete(1.0,tk.END)
            for i in range(min(5,len(X))): 
                self.x_input_text.insert(tk.END,",".join(map(str,X[i]))+ (";\n" if i<min(4,len(X)-1) else ""))
                y_d=Y[i]; y_s=str(y_d.index(1.0)) if self.loss_function_var.get()=="cross_entropy" and isinstance(y_d,list) and 1.0 in y_d else (",".join(map(str,y_d)) if isinstance(y_d,list) else str(y_d))
                self.y_input_text.insert(tk.END,y_s + (";\n" if i<min(4,len(Y)-1) else ""))
            self.log_message(f"{len(X)} örnek CSV'den yüklendi: {fp}")
            if not X: messagebox.showwarning("Veri Yükleme","CSV'den geçerli örnek yüklenemedi.",parent=self.master)
        except Exception as e: messagebox.showerror("CSV Okuma Hatası",f"CSV okunurken: {e}",parent=self.master); self.training_data_X,self.training_data_Y=[],[]; import traceback; traceback.print_exc()

    def save_canvas_as_eps(self):
        if not self.network or not self.network.weights: messagebox.showinfo("Bilgi","Kaydedilecek ağ görseli yok.",parent=self.master); return
        fp=filedialog.asksaveasfilename(title="Ağ Görselini Kaydet",defaultextension=".eps",filetypes=(("Encapsulated PostScript","*.eps"),("Tüm Dosyalar","*.*")),parent=self.master)
        if not fp: return
        try: self.canvas.postscript(colormode='color',file=fp); self.log_message(f"Ağ görseli EPS olarak kaydedildi: {fp}"); messagebox.showinfo("Başarılı",f"Ağ görseli {fp} adresine kaydedildi.\nEPS'yi PNG'ye dönüştürmek için Ghostscript veya online araçlar kullanabilirsiniz.",parent=self.master)
        except Exception as e: messagebox.showerror("Kaydetme Hatası",f"Görsel kaydedilirken: {e}",parent=self.master)

    def save_network_with_state(self):
        if not self.network or not self.network.weights: messagebox.showinfo("Bilgi","Kaydedilecek ağ yok.",parent=self.master); return
        fp=filedialog.asksaveasfilename(title="Ağı ve Eğitim Durumunu Kaydet",defaultextension=".json",filetypes=(("JSON Dosyaları","*.json"),("Tüm Dosyalar","*.*")),parent=self.master)
        if not fp: return
        state_data={"input_size":self.input_size_var.get(),"layer_configs_full":self.network.layer_configs,"weights":self.network.weights,"biases":self.network.biases,"loss_function":self.loss_function_var.get(),
            "training_state":{"epoch_losses":self.current_epoch_losses,"epoch_accuracies":self.current_epoch_accuracies,"total_epochs_completed":self.epochs_var.get(),
                              "optimizer_adam_t":self.network.adam_t, "optimizer_velocity_W": self.network.velocity_W, "optimizer_velocity_b":self.network.velocity_b,
                              "optimizer_m_W":self.network.m_W, "optimizer_v_W":self.network.v_W, "optimizer_m_b":self.network.m_b, "optimizer_v_b":self.network.v_b},
            "optimizer_state":{"type":self.optimizer_var.get(),"adam_t":self.network.adam_t,"velocity_W":self.network.velocity_W,"velocity_b":self.network.velocity_b,"m_W":self.network.m_W,"v_W":self.network.v_W,"m_b":self.network.m_b,"v_b":self.network.v_b}}
        try: 
            with open(fp,'w') as f: json.dump(state_data,f,indent=2); self.log_message(f"Ağ ve eğitim durumu kaydedildi: {fp}")
        except Exception as e: 
            messagebox.showerror("Kaydetme Hatası",f"Ağ kaydedilirken: {e}",parent=self.master)

    def load_network_with_state(self):
        fp=filedialog.askopenfilename(title="Ağ ve Eğitim Durumunu Yükle (.json)",filetypes=(("JSON","*.json"),("Tüm Dosyalar","*.*")),parent=self.master)
        if not fp: return
        try:
            with open(fp,'r') as f: data=json.load(f)
            self.input_size_var.set(data["input_size"]); self.loss_function_var.set(data.get("loss_function","mean_squared_error"))
            training_state_loaded=data.get("training_state")
            self.build_and_draw_network(data["weights"],data["biases"],data.get("layer_configs_full",data.get("layer_configs")),training_state=training_state_loaded)
            opt_state=data.get("optimizer_state") 
            if not opt_state and training_state_loaded : opt_state={key.replace("optimizer_",""):val for key,val in training_state_loaded.items() if key.startswith("optimizer_")}
            
            if opt_state and self.network: 
                self.optimizer_var.set(opt_state.get("type","sgd")); self.network.adam_t=opt_state.get("adam_t",0)
                for attr in ["velocity_W","velocity_b","m_W","v_W","m_b","v_b"]:
                    val=opt_state.get(attr)
                    if val is not None and hasattr(self.network,attr) and len(val)==len(getattr(self.network,attr)): setattr(self.network,attr,val)
                self.log_message("Optimizer durumu da yüklendi.")
            self.log_message(f"Ağ ve eğitim durumu yüklendi: {fp}")
        except Exception as e: messagebox.showerror("Yükleme Hatası",f"Ağ yüklenirken: {e}",parent=self.master); import traceback; traceback.print_exc()