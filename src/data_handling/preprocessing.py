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

def get_smooth_penalty(length, smoothness):
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2))
    return smoothness * D.dot(D.transpose())

COMMON_SMOOTH_PENALTY = get_smooth_penalty(INTERPOLATION_N_POINTS, BASELINE_SMOOTHNESS)
TARGET_X = np.linspace(CROP_MIN_X, CROP_MAX_X, INTERPOLATION_N_POINTS)

def preprocess_samples(samples):
    samples_length_pre = len(samples)

    pbar = tqdm(total=samples_length_pre, desc="Preprocessing samples", colour="yellow")

    # process first sample with visualization steps
    preprocess_with_plot(samples[0])

    # Process the rest and filter empty samples
    for sample in samples:
        preprocess_sample(sample)
        pbar.update(1)

    # delete invalid samples
    samples[:] = [s for s in samples if len(s.y) == INTERPOLATION_N_POINTS]

    pbar.close()

    rprint(f"[bold green]Preprocessed {len(samples)}/{samples_length_pre} valid samples.[/bold green]")

def preprocess_with_plot(sample):
    fig, axs = get_subplots()

    plot_sample(sample, axs, 0, "Original")

    # Crop to fingerprint region
    crop_spectrum(sample, CROP_MIN_X, CROP_MAX_X)
    plot_sample(sample, axs, 1, "Cropped")

    # Check for empty spectrum
    if (len(sample.x) == 0):
        return fig, axs
    
    # Interpolate to fixed length
    interpolate_spectrum(sample)
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
    crop_spectrum(sample, CROP_MIN_X, CROP_MAX_X)

    # Check for empty spectrum
    if (len(sample.x) == 0):
        return
    
    # Interpolate to fixed length
    interpolate_spectrum(sample)

    # Baseline correction
    correct_baseline(sample, smoothness = BASELINE_SMOOTHNESS, asymmetry = BASELINE_ASMMETRY, max_iterations = BASELINE_MAX_ITERATIONS)
    
    # Smooth the spectrum (for taking out noise)
    smooth_spectrum(sample, window_length = SMOOTHING_WINDOW_LENGTH, polyorder = SMOOTHING_POLYORDER)

    # Normalize intensities
    normalize_spectrum(sample)


def crop_spectrum(sample, min_x, max_x):
    left = np.searchsorted(sample.x, min_x, side='left')
    right = np.searchsorted(sample.x, max_x, side='right')
    sample.x = sample.x[left:right]
    sample.y = sample.y[left:right]


def interpolate_spectrum(sample):
    f = interp1d(sample.x, sample.y, kind='linear', bounds_error=False, fill_value=0)
    sample.x = TARGET_X
    sample.y = f(TARGET_X)


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

    # Initial weights: all equal
    weights = np.ones(length)

    for _ in range(max_iterations):
        W = sparse.spdiags(weights, 0, length, length)
        Z = W + COMMON_SMOOTH_PENALTY
        baseline = spsolve(Z, weights * signal)

        # Update weights: lower for points above the baseline (likely peaks)
        weights = np.where(signal > baseline, asymmetry, 1 - asymmetry)

    # Subtract the estimated baseline from the original signal
    sample.y = signal - baseline
    return sample


def smooth_spectrum(sample, window_length, polyorder):
    # Apply Savitzky-Golay smoothing to reduce noise.
    sample.y = savgol_filter(sample.y, window_length=window_length, polyorder=polyorder)

def normalize_spectrum(sample):
    # Min-Max normalization to scale intensities between 0 and 1.
    range_y = np.ptp(sample.y)
    if range_y == 0:
        sample.y = np.array([]) # set y array to empty so sample is skipped
    else:
        sample.y = (sample.y - np.min(sample.y)) / range_y
