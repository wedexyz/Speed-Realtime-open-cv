import numpy as np
from numba import jit


@jit(nopython=True)
def compare_ssim(image1, image2):
    k1 = 0.01
    k2 = 0.03
    win_size = 7

    # ndimage filters need floating point data
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    # compute (weighted) means
    ux = weighted_mean(image1, win_size)
    uy = weighted_mean(image2, win_size)

    # compute (weighted) variances and covariances
    vx = variance(image1, win_size)
    vy = variance(image2, win_size)

    vxy = covariance(image1, image2, win_size, ux, uy)

    c1 = (k1 * 255) ** 2
    c2 = (k2 * 255) ** 2

    ssim_values = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))

    return np.mean(ssim_values), ssim_values


@jit(nopython=True)
def weighted_mean(image, win_size):
    height_image = len(image)
    width_image = len(image[0])
    result = np.zeros_like(image)

    for y in range(height_image):
        for x in range(width_image):
            y_start = y - int(win_size / 2)
            y_end = y_start + win_size
            x_start = x - int(win_size / 2)
            x_end = x_start + win_size

            if y_start < 0:
                y_start = 0
            if y_end >= height_image:
                y_end = height_image - 1
            if x_start < 0:
                x_start = 0
            if x_end >= width_image:
                x_end = width_image - 1

            result[y, x] = np.mean(image[y_start:y_end, x_start:x_end])

    return result


@jit(nopython=True)
def variance(image, win_size):
    height_image = len(image)
    width_image = len(image[0])
    result = np.zeros_like(image)

    for y in range(height_image):
        for x in range(width_image):
            y_start = y - int(win_size / 2)
            y_end = y_start + win_size
            x_start = x - int(win_size / 2)
            x_end = x_start + win_size

            if y_start < 0:
                y_start = 0
            if y_end >= height_image:
                y_end = height_image - 1
            if x_start < 0:
                x_start = 0
            if x_end >= width_image:
                x_end = width_image - 1

            result[y, x] = np.var(image[y_start:y_end, x_start:x_end])

    return result


@jit
def covariance(image1, image2, win_size, mean1, mean2):
    height_image = len(image1)
    width_image = len(image1[0])
    result = np.zeros_like(image1)

    for y in range(height_image):
        for x in range(width_image):
            y_start = y - int(win_size / 2)
            y_end = y_start + win_size
            x_start = x - int(win_size / 2)
            x_end = x_start + win_size

            if y_start < 0:
                y_start = 0
            if y_end >= height_image:
                y_end = height_image - 1
            if x_start < 0:
                x_start = 0
            if x_end >= width_image:
                x_end = width_image - 1

            for row in range(y_start, y_end):
                for col in range(x_start, x_end):
                    result[y, x] += (image1[row, col] - mean1[y, x]) * (image2[row, col] - mean2[y, x])

            result[y, x] = result[y, x] / ((y_end - y_start) * (x_end - x_start) - 1)

    return result
