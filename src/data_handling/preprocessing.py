import numpy as np
from data_handling.visualizer import plot_sample, get_subplots
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from rich import print as rprint

from config import (
    CROP_MIN_X, CROP_MAX_X, INTERPOLATION_N_POINTS,
    SMOOTHING_WINDOW_LENGTH, SMOOTHING_POLYORDER,
    BASELINE_SMOOTHNESS, BASELINE_ASMMETRY, BASELINE_MAX_ITERATIONS
)

def preprocess_samples(samples):
    samples_length_pre = len(samples)

    pbar = tqdm(total=samples_length_pre, desc="Preprocessing samples", colour="yellow")

    # process first sample with visualization steps
    preprocess_with_plot(samples[0])

    # Process the rest and filter empty samples
    write_index = 0
    for read_index in range(len(samples)):
        sample = samples[read_index]
        preprocess_sample(sample)
        if len(sample.y) == INTERPOLATION_N_POINTS:
            samples[write_index] = sample
            write_index += 1
        pbar.update(1)

    # delete everything past last sample
    del samples[write_index:]
    pbar.close()

    rprint(f"[bold green]Preprocessed {len(samples)}/{samples_length_pre} valid samples.[/bold green]")

def preprocess_with_plot(sample):
    fig, axs = get_subplots()

    plot_sample(sample, axs, 0, "Original")

    # Crop to fingerprint region
    crop_spectrum(sample, min_x = CROP_MIN_X, max_x = CROP_MAX_X)
    plot_sample(sample, axs, 1, "Cropped")

    # Check for empty spectrum
    if (len(sample.x) == 0):
        return fig, axs
    
    # Interpolate to fixed length
    target_x = np.linspace(CROP_MIN_X, CROP_MAX_X, INTERPOLATION_N_POINTS)
    interpolate_spectrum(sample, target_x)
    plot_sample(sample, axs, 2, "Interpolated")

    # Baseline correction
    correct_baseline(sample, smoothness = BASELINE_SMOOTHNESS, asymmetry = BASELINE_ASMMETRY, max_iterations = BASELINE_MAX_ITERATIONS)
    plot_sample(sample, axs, 3, "Baseline correction")
    
    # Smooth the spectrum (for taking out noise)
    smooth_spectrum(sample, window_length = SMOOTHING_WINDOW_LENGTH, polyorder = SMOOTHING_POLYORDER)
    plot_sample(sample, axs, 5, "Smoothing")

    # Normalize intensities
    normalize_spectrum(sample)
    plot_sample(sample, axs, 4, "Normalized")

    return fig, axs

def preprocess_sample(sample):
    # Crop to fingerprint region
    crop_spectrum(sample, min_x = CROP_MIN_X, max_x = CROP_MAX_X)

    # Check for empty spectrum
    if (len(sample.x) == 0):
        return
    
    # Interpolate to fixed length
    target_x = np.linspace(CROP_MIN_X, CROP_MAX_X, INTERPOLATION_N_POINTS)
    interpolate_spectrum(sample, target_x)

    # Baseline correction
    correct_baseline(sample, smoothness = BASELINE_SMOOTHNESS, asymmetry = BASELINE_ASMMETRY, max_iterations = BASELINE_MAX_ITERATIONS)
    
    # Smooth the spectrum (for taking out noise)
    smooth_spectrum(sample, window_length = SMOOTHING_WINDOW_LENGTH, polyorder = SMOOTHING_POLYORDER)

    # Normalize intensities
    normalize_spectrum(sample)


def crop_spectrum(sample, min_x, max_x):
    mask = (sample.x >= min_x) & (sample.x <= max_x)
    sample.x = sample.x[mask]
    sample.y = sample.y[mask]


def interpolate_spectrum(sample, target_x):
    f = interp1d(sample.x, sample.y, kind='linear', bounds_error=False, fill_value=0)
    sample.x = target_x
    sample.y = f(target_x)


def correct_baseline(sample, smoothness, asymmetry, max_iterations):
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

def normalize_spectrum(sample):
    # Min-Max normalization to scale intensities between 0 and 1.
    min_y = np.min(sample.y)
    max_y = np.max(sample.y)
    if max_y - min_y == 0:
        sample.y = np.array([]) # set y array to empty so sample is skipped
    else:
        sample.y = (sample.y - min_y) / (max_y - min_y)

def smooth_spectrum(sample, window_length, polyorder):
    # Apply Savitzky-Golay smoothing to reduce noise.
    sample.y = savgol_filter(sample.y, window_length=window_length, polyorder=polyorder)