import numpy as np
import scipy


def running_difference(time_space_arr: np.ndarray) -> np.ndarray:
    return scipy.ndimage.correlate(
        time_space_arr,
        np.array([[-1, 1, 0]]).T,
    )


def base_difference(time_space_arr: np.ndarray):
    t_len, x_len = time_space_arr.shape
    difference = np.empty((t_len, x_len), dtype=float)
    for t in range(t_len):
        difference[t] = time_space_arr[t] - time_space_arr[0]
    return difference


def boxcar_mask(time_space_arr: np.ndarray, t: int, x: int, **kwargs):
    return scipy.ndimage.correlate(
        time_space_arr,
        np.ones((t, x)) / float(t * x),
        **kwargs
    )


def gradient_mask(time_space_arr: np.ndarray, **kwargs):
    return scipy.ndimage.correlate(
        time_space_arr,
        np.array([[-1, 0, 1]]).T,
        **kwargs
    )


def sobel_mask(time_space_arr: np.ndarray, **kwargs):
    return scipy.ndimage.correlate(
        time_space_arr,
        np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]).T,
        **kwargs
    )
