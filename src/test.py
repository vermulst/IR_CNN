import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, Label, Button
import os

from cnn.CNN import BasicCNN1D
from data_handling.preprocessing import preprocess_samples
from data_handling.loader import read_spectra_samples_no_smiles
from config import INTERPOLATION_N_POINTS, FUNCTIONAL_GROUP_SMARTS

class SpectraApp:
    def __init__(self, root):
        self.class_names = list(FUNCTIONAL_GROUP_SMARTS.keys())

        # Load model
        self.model = BasicCNN1D(INTERPOLATION_N_POINTS, len(self.class_names))
        self.model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
        print("[DEBUG] Model loaded successfully.")
        print(f"[DEBUG] Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        self.model.eval()

        # UI
        root.title("Spectra CNN Classifier")
        root.geometry("500x400")

        Button(root, text="Select .jdx File", command=self.select_file).pack(pady=10)
        Button(root, text="Select Folder of .jdx Files", command=self.select_folder).pack(pady=10)
        self.result_label = Label(root, text="", wraplength=480, justify="left")
        self.result_label.pack(pady=20)

    def predict(self, sample):
        y = torch.tensor(sample.y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = self.model(y).squeeze(0)  # shape: (num_classes,)
            print(f"[DEBUG] Raw logits: {out.numpy()}")  # <- add this
            probs = out.sigmoid().numpy()  # convert logits to probabilities
            print(f"[DEBUG] Sigmoid probs: {probs}")  # <- add this

        results = [(name, prob) for name, prob in zip(self.class_names, probs)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def process_paths(self, paths):
        samples = read_spectra_samples_no_smiles(paths, dataset_type="Test")
        preprocess_samples(samples)
        results = []
        for sample in samples:
            if sample.skip:
                continue
            print(f"[DEBUG] {sample.path}")
            preds = self.predict(sample)
            lines = [f"{name}: {prob:.2f}" for name, prob in preds]
            result_str = f"{os.path.basename(sample.path)}:\n  " + "\n  ".join(lines)
            results.append(result_str)
        return results

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JCAMP-DX files", "*.dx *.jdx")])
        if file_path:
            results = self.process_paths([file_path])
            self.result_label.config(text="\n".join(results))

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            paths = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".dx", ".jdx"))
            ]
            results = self.process_paths(paths)
            self.result_label.config(text="\n".join(results[:20]))  # limit to 20 entries


def main():
    root = tk.Tk()
    app = SpectraApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
