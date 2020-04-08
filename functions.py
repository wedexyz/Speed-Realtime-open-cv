import numpy as np
from numba import jit


@jit(nopython=True)
def inverse_threshold(image, threshold):
    height_image = len(image)
    width_image = len(image[0])

    result = np.zeros_like(image)

    for y in range(height_image):
        for x in range(width_image):
            if image[y, x] < threshold:
                result[y, x] = 255

    return result
