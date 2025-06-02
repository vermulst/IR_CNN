import numpy as np
from data_handling.visualizer import plot_sample, get_subplots
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from config import (
    REGIONS, INTERPOLATION_N_POINTS,
    SMOOTHING_WINDOW_LENGTH, SMOOTHING_POLYORDER,
    BASELINE_SMOOTHNESS, BASELINE_ASMMETRY, BASELINE_MAX_ITERATIONS
)

# for printing
from tqdm import tqdm
from rich import print as rprint

# for optimizing
import os
import multiprocessing as mp
import copy


def get_smooth_penalty(length, smoothness):
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2))
    return smoothness * D.dot(D.transpose())

def get_target_region():
    total_length = sum([end - start for start, end in REGIONS])
    regions = [
        np.linspace(min_x, max_x, int(INTERPOLATION_N_POINTS * ((max_x - min_x) / total_length)))
        for min_x, max_x in REGIONS
    ]
    # Fix to ensure total points matches INTERPOLATION_N_POINTS exactly:
    total_allocated = sum(len(r) for r in regions)
    if total_allocated < INTERPOLATION_N_POINTS:
        # Add extra points to the last region
        last_region = regions[-1]
        extra_points = INTERPOLATION_N_POINTS - total_allocated
        regions[-1] = np.linspace(last_region[0], last_region[-1], len(last_region) + extra_points)

    return np.concatenate(regions)

COMMON_SMOOTH_PENALTY = get_smooth_penalty(INTERPOLATION_N_POINTS, BASELINE_SMOOTHNESS)
TARGET_X = get_target_region()


def preprocess_samples(samples):
    samples_length_pre = len(samples)

    # check for empty samples
    if samples_length_pre == 0:
        rprint("[bold red]No samples to process![/bold red]")
        return
    
    # process first sample with visualization steps
    preprocess_with_plot(samples[0])

    # preprocess samples in parallel
    with mp.Pool(calculate_max_workers()) as pool:
        processed = []
        for result in tqdm(pool.imap_unordered(process_wrapper, enumerate(samples[1:], start=1)),
                          total=len(samples) - 1,
                          desc="Processing samples",
                          colour="yellow"):
            if result is not None:
                idx, processed_sample = result
                samples[idx] = processed_sample
                processed.append(idx)
            
    # delete invalid samples
    valid_samples = [samples[0]]  # Keep plotted sample
    valid_samples.extend(samples[i] for i in sorted(processed) if i > 0)  # Add parallel-processed
    samples[:] = valid_samples

    rprint(f"[bold green]Preprocessed {len(samples)}/{samples_length_pre} valid samples.[/bold green]")

def process_wrapper(args):
    idx, sample = args
    try:
        # Create a copy to work on
        sample_copy = copy.deepcopy(sample)
        preprocess_sample(sample_copy)
        if len(sample_copy.y) == INTERPOLATION_N_POINTS:
            return (idx, sample_copy)
    except Exception:
        pass
    return None

def preprocess_with_plot(sample):
    fig, axs = get_subplots()

    plot_sample(sample, axs, 0, "Original")
    
    # Interpolate to fixed length
    interpolate_spectrum(sample)
    plot_sample(sample, axs, 2, "Interpolated")

    # Crop to fingerprint region
    crop_spectrum(sample, REGIONS)
    plot_sample(sample, axs, 1, "Cropped")

    # Check for empty spectrum
    if (len(sample.x) == 0):
        return fig, axs

    # Baseline correction
    #correct_baseline(sample, smoothness = BASELINE_SMOOTHNESS, asymmetry = BASELINE_ASMMETRY, max_iterations = BASELINE_MAX_ITERATIONS)
    plot_sample(sample, axs, 3, "Baseline correction")
    
    # Smooth the spectrum (for taking out noise)
    #smooth_spectrum(sample, window_length = SMOOTHING_WINDOW_LENGTH, polyorder = SMOOTHING_POLYORDER)
    plot_sample(sample, axs, 5, "Smoothing")

    # Normalize intensities
    normalize_spectrum(sample)
    plot_sample(sample, axs, 4, "Normalized")

    return fig, axs

def preprocess_sample(sample):
    # Interpolate to fixed length
    interpolate_spectrum(sample)

    # Crop to fingerprint region
    crop_spectrum(sample, REGIONS)

    # Check for empty spectrum
    if (len(sample.x) == 0):
        return

    # Baseline correction
    #correct_baseline(sample, smoothness = BASELINE_SMOOTHNESS, asymmetry = BASELINE_ASMMETRY, max_iterations = BASELINE_MAX_ITERATIONS)
    
    # Smooth the spectrum (for taking out noise)
    #smooth_spectrum(sample, window_length = SMOOTHING_WINDOW_LENGTH, polyorder = SMOOTHING_POLYORDER)

    # Normalize intensities
    normalize_spectrum(sample)

def interpolate_spectrum(sample):
    f = interp1d(sample.x, sample.y, kind='linear', bounds_error=False, fill_value=0)
    sample.x = TARGET_X
    sample.y = f(TARGET_X)

def crop_spectrum(sample, regions):
    x_segments = []
    y_segments = []

    for min_x, max_x in regions:
        left = np.searchsorted(sample.x, min_x, side='left')
        right = np.searchsorted(sample.x, max_x, side='right')
        x_segments.append(sample.x[left:right])
        y_segments.append(sample.y[left:right])

    sample.x = np.concatenate(x_segments)
    sample.y = np.concatenate(y_segments)


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

def calculate_max_workers():
    cpu_count = os.cpu_count() or 1
    return min(32, int(cpu_count / 2))
