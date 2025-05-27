import numpy as np


def saturation(saturation_value: float, vector: np.ndarray):
    abs_vec = abs(vector)
    sign = np.where(vector == 0, 0, vector / abs_vec)
    condition = abs_vec >= saturation_value
    measurements = np.where(condition, sign * saturation_value, vector)
    indices = {"saturated_indices": condition}
    return measurements, indices


def relu(bias: np.ndarray, vector: np.ndarray):
    condition = vector < 0
    unbiased = bias == 0
    measurements = np.where(condition, 0, vector) - bias
    indices = {"unbiased_indices": unbiased}
    return measurements, indices


def gate(threshold: float, vector: np.ndarray):
    condition = vector < threshold
    measurements = np.where(condition, 0, vector)
    indices = {
        "gated_indices": condition,
    }
    return measurements, indices


def band_removal_logan(
    lower_threshold: float, upper_threshold: float, vector: np.ndarray
):
    condition = lower_threshold <= vector & vector < upper_threshold
    measurements = np.where(condition, lower_threshold, vector)
    indices = {"in_band_indices": condition}
    return measurements, indices


def band_removal_brody(
    lower_threshold: float, upper_threshold: float, vector: np.ndarray
):
    condition1 = lower_threshold <= vector & vector < upper_threshold
    measurements = np.where(condition1, lower_threshold, vector)
    condition2 = measurements > lower_threshold
    measurements = np.where(condition2, vector - lower_threshold, vector)
    indices = {
        "below_band_indices": ~(condition1 & condition2),
        "in_band_indices": condition1,
        "above_band_indices": condition2,
    }
    return measurements, indices
