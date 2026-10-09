# Ana GUI sınıfını çağırarak arayüzü görüntüler.

import tkinter as tk
if __package__:
    from .gui import DeepLearningSimulatorGUI
else:
    from gui import DeepLearningSimulatorGUI

if __name__ == '__main__':
    root = tk.Tk()
    root.option_add("*Font", "Calibri 10") 
    
    app = DeepLearningSimulatorGUI(root)
    
    root.mainloop()
