import numpy as np

__all__ = ["jitter", "scaling", "permutation"]


def jitter(x, sigma: float = 0.03):
    return x + np.random.normal(0.0, sigma, x.shape)


def scaling(x, sigma: float = 0.1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[1],))
    return x * factor


def permutation(x, max_segments: int = 5):
    idx = np.arange(x.shape[0])
    segs = np.array_split(idx, np.random.randint(1, max_segments))
    np.random.shuffle(segs)
    return x[np.concatenate(segs)]