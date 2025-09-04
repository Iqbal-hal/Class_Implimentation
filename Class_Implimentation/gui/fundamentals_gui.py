import tkinter as tk

class FundamentalsWindow:
    def __init__(self, parent, initial_data=None, initial_scrip=""):
        # Create the Toplevel window linked to the parent.
        self.window = tk.Toplevel(parent)
        self.window.title(f"Fundamentals Metrics - {initial_scrip}")
        
        # Create labels for each fundamental metric.
        self.label_pe = tk.Label(self.window, text="P/E Ratio: N/A", font=("Arial", 12))
        self.label_pe.pack(pady=5)
        self.label_eps = tk.Label(self.window, text="EPS: N/A", font=("Arial", 12))
        self.label_eps.pack(pady=5)
        self.label_peg = tk.Label(self.window, text="PEG Ratio: N/A", font=("Arial", 12))
        self.label_peg.pack(pady=5)
        
        # If initial data is provided, update the window.
        if initial_data:
            self.update_data(initial_data, initial_scrip)
    
    def update_data(self, fundamental_data, scrip_name=""):
        """
        Update the labels and window title with new fundamental data.
        Expects fundamental_data to be a dict-like object with keys: 'PE', 'EPS', 'PEG'.
        """
        # Update window title to include the scrip name.
        self.window.title(f"Fundamentals Metrics - {scrip_name}")
        
        # Extract and update metrics.
        pe = fundamental_data.get("P/E", "N/A")
        eps = fundamental_data.get("EPS", "N/A")
        peg = fundamental_data.get("PEG", "N/A")
        self.label_pe.config(text=f"P/E Ratio: {pe}")
        self.label_eps.config(text=f"EPS: {eps}")
        self.label_peg.config(text=f"PEG Ratio: {peg}")
