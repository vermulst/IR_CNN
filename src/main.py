from data_handling.loader import load_samples
from data_handling.preprocessing import process_samples
from data_handling.visualizer import plot_show

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


if __name__ == "__main__":
    main()
