import matplotlib.pyplot as plt

def plot_sample(sample):
    plt.plot(sample.x, sample.y)
    plt.title("Raw Spectrum")
    plt.xlabel("Wavelength / cm⁻¹")
    plt.ylabel("Absorbance or Intensity")
    plt.show()