import numpy as np

from numba import jit


@jit
def find_centroid(contour):
    return np.mean(contour[:, 0][:, 0]), np.mean(contour[:, 0][:, 1])
