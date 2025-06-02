from data_handling.loader import load_samples
from data_handling.preprocessing import preprocess_samples
from data_handling.visualizer import plot_show

import torch
import torch.nn as nn

from cnn.CNN import BasicCNN1D
from cnn.CDataset import CustomArrayDataset
from torch.utils.data import DataLoader

from config import FUNCTIONAL_GROUP_SMARTS

import time


def main():
    # load samples
    samples = load_samples("data/public")

    # preprocess
    preprocess_samples(samples)


    # Print class distribution for ALL samples
    class_names = list(FUNCTIONAL_GROUP_SMARTS.keys())
    num_classes = len(class_names)
    class_counts = [0] * num_classes  # Track positives per class
    
    for sample in samples:
        labels = sample.labels  # Assuming labels is a list/array of 0s and 1s
        for i in range(num_classes):
            if labels[i] == 1:
                class_counts[i] += 1
    
    print("\nClass distribution (actual positives) in FULL dataset:")
    for i, name in enumerate(class_names):
        print(f"{name}: {class_counts[i]} ({(100 * class_counts[i] / len(samples)):.1f}%)")


    # CNN
    num_samples = len(samples)
    num_classes = len(samples[0].labels) 

    train_dataset = CustomArrayDataset(samples[:int(num_samples * 0.8)]) # 80% for training
    test_dataset = CustomArrayDataset(samples[int(num_samples * 0.8):])  # 20% for testing

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # printing CNN info
    input_length = max(len(s.y) for s in samples)  # Use maximum length
    print(f"Using input length: {input_length}")

    sample_input, sample_label = train_dataset[0]
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Sample label shape: {sample_label.shape}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicCNN1D(input_length=input_length, num_classes=num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 100
    best_accuracy = 0.0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation after each epoch
        model.eval()
        with torch.no_grad():
            correct = total = 0

            # Per-class metrics
            class_correct = [0, 0, 0, 0, 0]  # Correct predictions per class
            class_total = [0, 0, 0, 0, 0]     # Total samples per class
            class_tp = [0, 0, 0, 0, 0]        # True positives per class
            class_fp = [0, 0, 0, 0, 0]        # False positives per class
            class_fn = [0, 0, 0, 0, 0]        # False negatives per class

            for inputs, labels in test_loader:
                outputs = model(inputs.to(device))
                preds = (outputs > 0.5).float()

                # Overall accuracy
                correct += (preds == labels).sum().item()
                total += labels.numel()

                # Per-class accuracy
                for i in range(num_classes):  # Loop over each functional group
                    # Accuracy components
                    class_correct[i] += (preds[:, i] == labels[:, i]).sum().item()
                    class_total[i] += labels.shape[0]
            
                    # TP, FP, FN for F1
                    class_tp[i] += ((preds[:, i] == 1) & (labels[:, i] == 1)).sum().item()
                    class_fp[i] += ((preds[:, i] == 1) & (labels[:, i] == 0)).sum().item()
                    class_fn[i] += ((preds[:, i] == 0) & (labels[:, i] == 1)).sum().item()

        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%')
        
        class_names = list(FUNCTIONAL_GROUP_SMARTS.keys())
        for i in range(num_classes):
            precision = class_tp[i] / (class_tp[i] + class_fp[i]) if (class_tp[i] + class_fp[i]) > 0 else 0
            recall = class_tp[i] / (class_tp[i] + class_fn[i]) if (class_tp[i] + class_fn[i]) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f'{class_names[i]}:')
            print(f'  Accuracy: {100 * class_correct[i] / class_total[i]:.2f}%')
            print(f'  F1: {100 * f1:.2f}%')

        # Macro-average F1 (average of per-class F1)
        macro_f1 = sum([
            2 * (class_tp[i] / (class_tp[i] + class_fp[i])) * (class_tp[i] / (class_tp[i] + class_fn[i])) / 
            (class_tp[i] / (class_tp[i] + class_fp[i]) + class_tp[i] / (class_tp[i] + class_fn[i])) 
            if (class_tp[i] + class_fp[i]) > 0 and (class_tp[i] + class_fn[i]) > 0 else 0
            for i in range(num_classes)
        ]) / num_classes
        
        print(f'\nMacro-average F1: {100 * macro_f1:.2f}%')

        # Micro-average F1 (global TP/FP/FN)
        global_tp = sum(class_tp)
        global_fp = sum(class_fp)
        global_fn = sum(class_fn)
        micro_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
        micro_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        print(f'Micro-average F1: {100 * micro_f1:.2f}%')

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Training complete. Best validation accuracy: {best_accuracy:.2f}%')


if __name__ == "__main__":
    main()
