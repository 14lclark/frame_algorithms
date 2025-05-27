from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from frames.frame import Frame, RandomFrame


def random_unit_vector(dimension: int):
    """
    Generate a random unit vector.
    """
    vec = np.random.randn(dimension, 1)
    vec = vec / np.sqrt(np.sum(vec**2))
    return vec


def error(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def standard_frame_algorithm(
    frame: Frame,
    vector: np.ndarray,
    num_iters: int,
    activation_fn: Callable = lambda x: (x, None),
):
    y = [np.zeros_like(vector) for _ in range(num_iters + 1)]
    errors = [0 for _ in range(num_iters)]
    measurements, indices = activation_fn(frame.analysis(vector))
    Sv = frame.synthesis(measurements)
    alpha = 2 / (frame.A + frame.B)
    for k in range(num_iters):
        Sy_k = frame.synthesis(frame.analysis(y[k]))
        y[k + 1] = y[k] + alpha * (Sv - Sy_k)
        errors[k] = error(vector, y[k + 1])
    return y, errors


def conjugate_gradient_algorithm(
    frame: Frame,
    vector: np.ndarray,
    num_iters: int,
    activation_fn: Callable = lambda x: (x, None),
):
    # y is actually just an upside down lambda
    y = [0 for _ in range(num_iters + 2)]
    g = [np.zeros_like(vector) for _ in range(num_iters + 2)]
    r = [np.zeros_like(vector) for _ in range(num_iters + 2)]
    p = [np.zeros_like(vector) for _ in range(num_iters + 2)]
    Sp = [np.zeros_like(vector) for _ in range(num_iters + 2)]
    p_dot_Sp = [0 for _ in range(num_iters + 2)]
    errors = [0 for _ in range(num_iters + 2)]
    measurements = activation_fn(frame.analysis(vector))
    r[1] = p[1] = frame.synthesis(measurements)
    for k in range(1, num_iters + 1):
        Sp[k] = frame.frame_operator @ p[k]
        p_dot_Sp[k] = (p[k] * Sp[k]).sum()
        y[k] = (r[k] * p[k]).sum() / p_dot_Sp[k]
        g[k + 1] = g[k] + y[k] * p[k]
        r[k + 1] = r[k] - y[k] * Sp[k]
        p[k + 1] = (
            Sp[k]
            - (Sp[k] * Sp[k]).sum() / p_dot_Sp[k] * p[k]
            - (
                (Sp[k] * Sp[k - 1]).sum()
                / (
                    1 if (a := p_dot_Sp[k - 1]) == 0 else a
                )  # if denom is 0, make it 1 instead
                * p[k - 1]
            )
        )
        errors[k + 1] = error(vector, g[k + 1])
    return g[2:], errors[2:]


def chebyshev_algorithm(
    frame: Frame,
    vector: np.ndarray,
    num_iters: int,
    activation_fn: Callable = lambda x: (x, None),
):
    pass


if __name__ == "__main__":
    dim = 16
    iters = 20
    frame = RandomFrame(dimension=dim, num_frame_vecs=37, is_unit=True)
    print("A: ", frame.A, "\nB: ", frame.B)
    vector = random_unit_vector(dimension=dim)
    recover, error = standard_frame_algorithm(frame, vector, iters)
    plt.semilogy(np.arange(1, iters + 1), error, "-ob")
    plt.xlabel("Iteration")
    plt.ylabel("||x-y_k||")
    plt.show()
