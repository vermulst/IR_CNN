import matplotlib.pyplot as plt

# Create a figure with 6 vertical subplots and specified figure size
def get_subplots():
    return plt.subplots(6, 1, figsize=(10, 15))

# Plot a single sample on the specified subplot axis
def plot_sample(sample, axs, index, title):
    axs[index].plot(sample.x, sample.y)       # Plot x vs y data
    axs[index].set_title(title)               # Set the title for the subplot

# Finalize and display the plot layout
def plot_show():
    plt.xlabel("Wavelength / cm⁻¹")           # Set common x-axis label
    plt.ylabel("Absorbance or Intensity")     # Set common y-axis label
    plt.tight_layout()                        # Adjust layout to avoid overlap
    plt.show()                                # Display the plot
