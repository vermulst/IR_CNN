### main.py
from data_handling.loader import load_samples
from data_handling.preprocessing import preprocess_samples

from cnn.trainer import train_model  # New trainer module
from config import FUNCTIONAL_GROUP_SMARTS

def main():
    # Load and preprocess samples
    samples_chemotion = load_samples("data/public/chemotion", "chemotion")
    samples_nist = load_samples("data/public/nist_dataset", "nist")
    samples = samples_chemotion + samples_nist
    preprocess_samples(samples)

    # Handle class weights
    for sample in samples:
        if not any(sample.labels):
            sample.weight = 0.2

    # Print class distribution
    class_names = list(FUNCTIONAL_GROUP_SMARTS.keys())
    class_counts = [0] * len(class_names)
    for sample in samples:
        for i, label in enumerate(sample.labels):
            if label == 1:
                class_counts[i] += 1

    print("\nClass distribution in dataset:")
    for i, name in enumerate(class_names):
        print(f"{name}: {class_counts[i]} ({(100 * class_counts[i] / len(samples)):.1f}%)")

    # train and save model
    train_model(samples, class_names)

if __name__ == "__main__":
    main()