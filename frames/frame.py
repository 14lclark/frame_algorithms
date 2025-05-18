import numpy as np


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

        Returns a vector with shape (m, 1), where m is the number of frame vectors.
        """
        return self._frame_vectors.T @ vector

    def synthesis(self, vector: np.ndarray):
        """
        Synthesis operator.
        Vector must have shape (m, 1), where m is the number of frame vectors.

        Returns a vector with shape (n, 1), where n is the dimension of the vector space.
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
