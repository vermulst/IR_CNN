from data_handling.loader import load_samples
from data_handling.preprocessing import process_samples
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
    process_samples(samples)

    # visualize
    plot_show()

    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")

    print(samples[0].labels)

    input_length = 1000
    num_classes = 5
    num_samples = len(samples)

    train_dataset = CustomArrayDataset(samples[:int(num_samples * 0.8)]) # 80% for training
    test_dataset = CustomArrayDataset(samples[int(num_samples * 0.8):])  # 20% for testing

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    combined_length = len(samples[0].x) + len(samples[0].y)
    num_classes = len(samples[0].labels)

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
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print statistics
            if batch_idx % 10 == 9:  # Print every 10 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}')
                running_loss = 0.0
        
        # Validation after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()  # Threshold at 0.5 for binary classification
                total += labels.size(0) * labels.size(1)  # Total number of labels
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Training complete. Best validation accuracy: {best_accuracy:.2f}%')

    # CNN
    #cnn = BasicCNN1D(input_length=len(samples), num_classes=10)
    #cnn.train()


if __name__ == "__main__":
    main()
