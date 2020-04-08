import numpy as np

from numba import jit


@jit(nopython=True)
def opening(image, mask, iterations=1):
    result = image.copy()
    for i in range(iterations):
        result = erosion(result, mask)

    for i in range(iterations):
        result = dilation(result, mask)

    return result


@jit(nopython=True)
def closing(image, mask, iterations=1):
    result = image.copy()
    for i in range(iterations):
        result = dilation(result, mask)

    for i in range(iterations):
        result = erosion(result, mask)

    return result


@jit(nopython=True, parallel=True)
def erosion(image, mask):
    height_image = len(image)
    width_image = len(image[0])

    result = np.zeros_like(image)

    for y in range(height_image):
        for x in range(width_image):
            if fits(image, mask, (x, y)):
                result[y, x] = 255

    return result


@jit(nopython=True, parallel=True)
def dilation(image, mask):
    height_image = len(image)
    width_image = len(image[0])

    result = np.zeros_like(image)

    for y in range(height_image):
        for x in range(width_image):
            if hits(image, mask, (x, y)):
                result[y, x] = 255

    return result


@jit(nopython=True)
def fits(image, mask, center):
    height_mask = len(mask)
    width_mask = len(mask[0])
    height_image = len(image)
    width_image = len(image[0])

    for row in range(height_mask):
        for col in range(width_mask):
            row_offset = row - height_mask + round(height_mask / 2)
            col_offset = col - width_mask + round(width_mask / 2)
            target = (center[0] + col_offset, center[1] + row_offset)

            if 0 > target[0] or target[0] >= width_image or 0 > target[1] or target[1] >= height_image:
                continue

            if image[target[1], target[0]] != mask[row, col]:
                return False

    return True


@jit(nopython=True)
def hits(image, mask, center):
    height_mask = len(mask)
    width_mask = len(mask[0])
    height_image = len(image)
    width_image = len(image[0])

    for row in range(height_mask):
        for col in range(width_mask):
            row_offset = row - height_mask + round(height_mask / 2)
            col_offset = col - width_mask + round(width_mask / 2)
            target = (center[0] + col_offset, center[1] + row_offset)
            if 0 > target[0] or target[0] >= width_image or 0 > target[1] or target[1] >= height_image:
                continue

            if image[target[1], target[0]] == mask[row, col]:
                return True

    return False


@jit(nopython=True)
def convert_mask(mask):
    height_mask = len(mask)
    width_mask = len(mask[0])

    for y in range(height_mask):
        for x in range(width_mask):
            if mask[y, x] == 1:
                mask[y, x] = 255

    return mask
