import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, Label, Button
import os
from cnn.CNN import BasicCNN1D
from data_handling.loader import read_spectra_samples_no_smiles

# load best CNN
model = BasicCNN1D()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()


# Predict function
def predict(sample):
    y = torch.tensor(sample.y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, N]
    with torch.no_grad():
        out = model(y)
        pred = torch.argmax(out, dim=1).item()
    return pred

# File processing
def process_paths(paths):
    samples = read_spectra_samples_no_smiles(paths, dataset_type="Test")
    results = []
    for sample in samples:
        if sample.skip:
            continue
        pred = predict(sample)
        results.append(f"{os.path.basename(sample.path)} -> Class {pred}")
    return results


# UI handlers
def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("JCAMP-DX files", "*.dx *.jdx")]
    )
    if file_path:
        results = process_paths([file_path])
        result_label.config(text="\n".join(results))

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".dx", ".jdx"))
        ]
        results = process_paths(paths)
        result_label.config(text="\n".join(results[:20]))  # limit to 20 entries

# GUI
root = tk.Tk()
root.title("Spectra CNN Classifier")
root.geometry("500x400")

Button(root, text="Select .jdx File", command=select_file).pack(pady=10)
Button(root, text="Select Folder of .jdx Files", command=select_folder).pack(pady=10)
result_label = Label(root, text="", wraplength=480, justify="left")
result_label.pack(pady=20)

root.mainloop()