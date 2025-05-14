import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from BaselineRemoval import BaselineRemoval

def process(sample):
    #Full preprocessing pipeline for an IR spectrum.
    # Crop to fingerprint region
    crop_spectrum(sample, min_x=500, max_x=4000)

    if (len(sample.x) == 0):
        return sample
    # Interpolate to fixed length
    interpolate_spectrum(sample, n_points=1000)

    # Baseline correction
    #correct_baseline(sample)
    
    # Normalize intensities
    normalize_spectrum(sample)
    
    # Smooth the spectrum (for taking out noise)
    smooth_spectrum(sample)

    return sample


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

def correct_baseline(sample):
    # Remove baseline drift using Asymmetric Least Squares (ALS)
    baseObj = BaselineRemoval(sample.y)
    sample.y = baseObj.als()

def smooth_spectrum(sample, window_length=5, polyorder=2):
    # Apply Savitzky-Golay smoothing to reduce noise.
    sample.y = savgol_filter(sample.y, window_length=window_length, polyorder=polyorder)