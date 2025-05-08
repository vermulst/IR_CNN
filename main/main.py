from loader import load_samples
from visualizer import plot_sample
import preprocessing as pp
import time

start = time.time()


# loading the samples from the data in correct format
samples = load_samples("data/public")
print(f"Loaded {len(samples)} valid samples.")

# visualize the first sample
plot_sample(samples[0])


end = time.time()
print(end - start)
