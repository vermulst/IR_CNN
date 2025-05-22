import numpy as np
from visualizer import plot_sample, get_subplots
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve


def process_samples(samples):
    # process first sample with visualization steps
    process_with_plot(samples[0])

    # process the rest normally
    for sample in samples[1:]:
        process(sample)
    print("finished processing")

def process_with_plot(sample):

    fig, axs = get_subplots()

    #Full preprocessing pipeline for an IR spectrum.
    # Crop to fingerprint region
    plot_sample(sample, axs, 0, "Original")

    crop_spectrum(sample, min_x=500, max_x=4000)

    plot_sample(sample, axs, 1, "Cropped")

    if (len(sample.x) == 0):
        return fig, axs
    # Interpolate to fixed length
    interpolate_spectrum(sample, n_points=1000)

    plot_sample(sample, axs, 2, "Interpolated")

    # Baseline correction
    correct_baseline(sample)

    plot_sample(sample, axs, 3, "Baseline correction")
    
    # Normalize intensities
    normalize_spectrum(sample)

    plot_sample(sample, axs, 4, "Normalized")
    
    # Smooth the spectrum (for taking out noise)
    smooth_spectrum(sample)

    plot_sample(sample, axs, 5, "Smoothing")

    return fig, axs

def process(sample):
    #Full preprocessing pipeline for an IR spectrum.
    # Crop to fingerprint region
    crop_spectrum(sample, min_x=500, max_x=4000)

    if (len(sample.x) == 0):
        return
    # Interpolate to fixed length
    interpolate_spectrum(sample, n_points=1000)

    # Baseline correction
    correct_baseline(sample)
    
    # Normalize intensities
    normalize_spectrum(sample)
    
    # Smooth the spectrum (for taking out noise)
    smooth_spectrum(sample)


def crop_spectrum(sample, min_x=500, max_x=4000):
    mask = (sample.x >= min_x) & (sample.x <= max_x)
    sample.x = sample.x[mask]
    sample.y = sample.y[mask]


def interpolate_spectrum(sample, n_points=1000):
    f = interp1d(sample.x, sample.y, kind='linear')
    sample.x = np.linspace(min(sample.x), max(sample.x), n_points)
    sample.y = f(sample.x)

def normalize_spectrum(sample):
    # Min-Max normalization to scale intensities between 0 and 1.
    sample.y = (sample.y - np.min(sample.y)) / (np.max(sample.y) - np.min(sample.y))


def correct_baseline(sample, smoothness=1e6, asymmetry=0.01, max_iterations=10):
    """
    Applies baseline correction using Asymmetric Least Squares (ALS) smoothing.

    Parameters:
    - smoothness: Controls how smooth the estimated baseline should be (lambda).
                  Higher values = smoother baseline.
    - asymmetry:  Asymmetry parameter (p), between 0 and 1.
                  Lower values assume that most data points are above the baseline.
    - max_iterations: Number of iterations to refine the baseline estimate.
    """
    signal = sample.y
    length = len(signal)

    # Second-order difference matrix for smoothing
    difference_matrix = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2))
    smooth_penalty = smoothness * difference_matrix.dot(difference_matrix.transpose())

    # Initial weights: all equal
    weights = np.ones(length)

    for _ in range(max_iterations):
        W = sparse.spdiags(weights, 0, length, length)
        Z = W + smooth_penalty
        baseline = spsolve(Z, weights * signal)

        # Update weights: lower for points above the baseline (likely peaks)
        weights = asymmetry * (signal > baseline) + (1 - asymmetry) * (signal < baseline)

    # Subtract the estimated baseline from the original signal
    sample.y = signal - baseline
    return sample

def smooth_spectrum(sample, window_length=5, polyorder=2):
    # Apply Savitzky-Golay smoothing to reduce noise.
    sample.y = savgol_filter(sample.y, window_length=window_length, polyorder=polyorder)