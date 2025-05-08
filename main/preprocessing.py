def process(sample):
    crop_spectrum(sample)
    return sample


def crop_spectrum(sample, min_x=500, max_x=4000):
    mask = (sample.x >= min_x) & (sample.x <= max_x)
    sample.x = sample.x[mask]
    sample.y = sample.y[mask]