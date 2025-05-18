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


def standard_frame_algorithm(frame: Frame, vector: np.ndarray, num_iters: int):
    y = [np.zeros_like(vector) for _ in range(num_iters + 1)]
    errors = [0 for _ in range(num_iters)]
    Sv = frame.synthesis(frame.analysis(vector))
    alpha = 2 / (frame.A + frame.B)
    for k in range(num_iters):
        Sy_k = frame.synthesis(frame.analysis(y[k]))
        y[k + 1] = y[k] + alpha * (Sv - Sy_k)
        errors[k] = np.sqrt(np.sum((vector - y[k + 1]) ** 2))
    return y, errors


def conjugate_gradient_algorithm(frame: Frame, vector: np.ndarray, num_iters: int):
    pass


if __name__ == "__main__":
    dim = 16
    iters = 20
    frame = RandomFrame(dimension=dim, num_frame_vecs=37, is_unit=True)
    vector = random_unit_vector(dimension=dim)
    recover, error = standard_frame_algorithm(frame, vector, iters)

    plt.semilogy(np.arange(1, iters + 1), error, "-ob")
    plt.xlabel("Iteration")
    plt.ylabel("||x-y_k||")
    plt.show()
