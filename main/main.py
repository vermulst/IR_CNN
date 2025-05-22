from loader import load_samples, load_first_sample
from visualizer import plot_sample, plot_show, get_subplots
from preprocessing import process, process_with_plot
import time

start = time.time()

# loading the samples from the data in correct format
samples = load_samples("data/public")
print(f"Loaded {len(samples)} valid samples.")

# preprocessing of data
fig, axs = get_subplots()
plot_sample(samples[0], axs, 0, "Original")

# process first sample with visualization steps
process_with_plot(samples[0], axs, 1)

# process the rest normally
for sample in samples[1:]:
    process(sample)

print("finished processing")

end = time.time()
print(end - start)

# visualize the first sample
plot_show()