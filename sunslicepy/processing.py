import numpy as np
import scipy


def running_difference(time_space_arr: np.ndarray) -> np.ndarray:
    t_len, x_len = time_space_arr.shape
    difference = np.empty((t_len - 1, x_len), dtype=float)
    for t in range(t_len - 1):
        for i in range(x_len):
            difference[t][i] = time_space_arr[t + 1][i] - time_space_arr[t][i]
    return difference


def boxcar_filter(
        time_space_arr: np.ndarray,
        t: int,
        x: int,
        **kwargs,
):
    return scipy.ndimage.convolve(
        time_space_arr,
        np.ones((t, x)) / float(t * x)
    )


def gradient_filter(time_space_arr: np.ndarray):
    """
        [-1  0  1] * I
    """

    raise NotImplemented


def sobel_filter(time_space_arr: np.ndarray):
    """
        [-1  0  1]
        [-2  0  2] * I
        [-1  0  1]
    """

    raise NotImplemented
