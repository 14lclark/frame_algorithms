import numpy as np


def saturation(saturation_value: float, vector: np.ndarray):
    abs_vec = abs(vector)
    sign = np.where(vector == 0, 0, vector / abs_vec)
    return np.where(abs_vec >= saturation_value, sign * saturation_value, vector)


def relu(bias: np.ndarray, vector: np.ndarray):
    return np.where(vector >= 0, vector, 0) - bias


def gate(threshold: float, vector: np.ndarray):
    return np.where(vector < threshold, 0, vector)


def arbitray_range_gate(
    lower_threshold: float, upper_threshold: float, vector: np.ndarray
):
    return np.where(
        lower_threshold <= vector & vector < upper_threshold, lower_threshold, vector
    )
