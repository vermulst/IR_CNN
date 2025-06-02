from data_handling.loader import load_samples
from data_handling.preprocessing import preprocess_samples
from data_handling.visualizer import plot_show

import torch
import torch.nn as nn

from cnn.CNN import BasicCNN1D
from cnn.CDataset import CustomArrayDataset
from torch.utils.data import DataLoader

import time


def main():
    start = time.time()

    # load samples
    samples = load_samples("data/public")

    # preprocess
    preprocess_samples(samples)

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
    num_epochs = 10
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
            for inputs, labels in test_loader:
                outputs = model(inputs.to(device))
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Training complete. Best validation accuracy: {best_accuracy:.2f}%')


if __name__ == "__main__":
    main()
