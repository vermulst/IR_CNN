from loader import load_samples
from visualizer import plot_show
from preprocessing import process_samples
import time

# measure time
start = time.time()



# load samples
samples = load_samples()

# preprocess
process_samples(samples)

# visualize
plot_show()



# measure time
end = time.time()
print(end - start)
