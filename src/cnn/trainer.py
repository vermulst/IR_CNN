import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from cnn.CNN import BasicCNN1D
from cnn.CDataset import CustomArrayDataset

def train_model(samples, class_names):
    num_classes = len(class_names)
    input_length = max(len(s.y) for s in samples)

    # Dataset split
    num_samples = len(samples)
    train_dataset = CustomArrayDataset(samples[:int(num_samples * 0.8)])
    test_dataset = CustomArrayDataset(samples[int(num_samples * 0.8):])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicCNN1D(input_length=input_length, num_classes=num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Variables for model measureing model performance
    best_macro_f1 = 0.0
    f1_history = [[] for _ in range(num_classes)]
    macro_f1_history = []
    loss_history = []

    os.makedirs("plots", exist_ok=True)

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for inputs, labels, weights in train_loader:
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = (criterion(outputs, labels) * weights.unsqueeze(1)).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            class_tp = [0] * num_classes
            class_fp = [0] * num_classes
            class_fn = [0] * num_classes
            class_correct = [0] * num_classes
            class_total = [0] * num_classes

            for inputs, labels, weights in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = (outputs > 0.5).float()
                for i in range(num_classes):
                    class_correct[i] += (preds[:, i] == labels[:, i]).sum().item()
                    class_total[i] += labels.size(0)
                    class_tp[i] += ((preds[:, i] == 1) & (labels[:, i] == 1)).sum().item()
                    class_fp[i] += ((preds[:, i] == 1) & (labels[:, i] == 0)).sum().item()
                    class_fn[i] += ((preds[:, i] == 0) & (labels[:, i] == 1)).sum().item()

        f1_scores = []
        for i in range(num_classes):
            precision = class_tp[i] / (class_tp[i] + class_fp[i]) if (class_tp[i] + class_fp[i]) else 0
            recall = class_tp[i] / (class_tp[i] + class_fn[i]) if (class_tp[i] + class_fn[i]) else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            f1_scores.append(f1)
            f1_history[i].append(f1)

        macro_f1 = sum(f1_scores) / num_classes
        macro_f1_history.append(macro_f1)

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Macro F1: {macro_f1:.4f}")
        for i in range(num_classes):
            acc = 100 * class_correct[i] / class_total[i] if class_total[i] else 0
            print(f"  {class_names[i]} - Acc: {acc:.2f}%, F1: {100 * f1_scores[i]:.2f}%")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), "best_model.pth")
            with open("best_model.txt", "w") as f:
                f.write(f"Epoch: {epoch + 1}\nMacro F1: {100 * macro_f1:.2f}%\n")
                for i, name in enumerate(class_names):
                    f.write(f"{name}: F1: {100 * f1_scores[i]:.2f}%\n")

    save_f1_plot(f1_history, class_names, out_path="plots/f1_per_class.png")
    save_macro_f1_plot(macro_f1_history, out_path="plots/macro_f1.png")
    save_loss_plot(loss_history, out_path="plots/loss.png")
    save_all_f1_with_macro_plot(f1_history, macro_f1_history, class_names, out_path="plots/all_f1_with_macro.png")

    print(f"Training complete. Best Macro F1: {100 * best_macro_f1:.2f}%")


def save_f1_plot(f1_history, class_names, out_path):
    plt.figure(figsize=(12, 6))
    for i, f1s in enumerate(f1_history):
        plt.plot(f1s, label=class_names[i])
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score per Functional Group over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_macro_f1_plot(macro_f1_history, out_path):
    plt.figure()
    plt.plot(macro_f1_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Macro F1 Score over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_loss_plot(loss_history, out_path):
    plt.figure()
    plt.plot(loss_history, color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_all_f1_with_macro_plot(f1_history, macro_f1_history, class_names, out_path):
    plt.figure(figsize=(12, 6))
    for i, f1s in enumerate(f1_history):
        plt.plot(f1s, label=class_names[i])
    plt.plot(macro_f1_history, label="Macro F1", color="black", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Scores per Functional Group with Macro F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
