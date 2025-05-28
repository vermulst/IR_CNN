import matplotlib.pyplot as plt

def get_subplots():
    return plt.subplots(6, 1, figsize=(10, 15))

def plot_sample(sample, axs, index, title):
    axs[index].plot(sample.x, sample.y)
    axs[index].set_title(title)

def plot_show():
    plt.xlabel("Wavelength / cm⁻¹")
    plt.ylabel("Absorbance or Intensity")
    plt.tight_layout()
    plt.show()