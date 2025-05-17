import numpy as np
from matplotlib import pyplot as plt


class Frame:
    """
    Class containing important frame information and methods.
    """

    def __init__(self, frame_vectors: np.ndarray):
        """
        For a frame in an n-dimensional vector space V with m frame vectors,
        frame_vectors should have shape (n, m).
        """
        self.vector_space_dim = frame_vectors.shape[0]
        self.number_of_frame_vectors = frame_vectors.shape[1]
        self._frame_vectors = frame_vectors
        self.frame_operator = frame_vectors @ frame_vectors.T
        self.A, self.B = self.calculate_frame_bounds()

    def calculate_frame_bounds(self):
        """
        Returns (A, B), the lower and upper frame bounds.
        """
        eigenvalues = np.linalg.eigvalsh(self.frame_operator)
        return eigenvalues[0], eigenvalues[-1]

    def analysis(self, vector: np.ndarray):
        """
        Analysis operator.
        Vector must have shape (n, 1), where n is the dimension of the vector space.

        Returns a vector with shape (m,1), where m is the number of frame vectors.
        """
        return self._frame_vectors.T @ vector

    def synthesis(self, vector: np.ndarray):
        """
        Synthesis operator.
        Vector must have shape (m, 1), where m is the number of frame vectors.

        Returns a vector with shape (n,1), where n is the dimension of the vector space.
        """
        return self._frame_vectors @ vector


class RandomFrame(Frame):
    """
    Class for creating and using random frames.
    """

    def __init__(self, dimension: int, num_frame_vecs: int, is_unit: bool = True):
        """
        If is_unit is true, scale the random frame vectors to be unit length.
        """
        frame = np.random.randn(dimension, num_frame_vecs)
        if is_unit:
            for col in range(num_frame_vecs):
                frame[:, col] = frame[:, col] / np.sqrt(np.sum(frame[:, col] ** 2))
        super().__init__(frame_vectors=frame)


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
        Syk = frame.synthesis(frame.analysis(y[k]))
        y[k + 1] = y[k] + alpha * (Sv - Syk)
        errors[k] = np.sqrt(np.sum((vector - y[k + 1]) ** 2))
    return y, errors


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
