from data_handling.loader import load_samples
from data_handling.preprocessing import process_samples
from data_handling.visualizer import plot_show

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

    print(samples[0].label)
    #input_length = 1000
    #num_classes = 5
    #num_samples = len(samples)

    #train_dataset = CustomArrayDataset(samples[:int(num_samples * 0.8)]) # 80% for training
    #test_dataset = CustomArrayDataset(samples[int(num_samples * 0.8):])  # 20% for testing

    #batch_size = 32
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #print(f"Number of training batches: {len(train_loader)}")
    #print(f"Number of testing batches: {len(test_loader)}")


    # CNN
    #cnn = BasicCNN1D(input_length=len(samples), num_classes=10)
    #cnn.train()


if __name__ == "__main__":
    main()
