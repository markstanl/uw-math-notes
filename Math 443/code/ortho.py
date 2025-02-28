import numpy as np

def is_orthogonal_matrix(A: np.ndarray) -> bool:
    """
    Verify that a matrix is orthogonal.

    :param A: the matrix
    :return: True if the matrix is orthogonal, False otherwise
    """
    if A.shape[0] != A.shape[1]:
        return False
    return np.allclose(A.T @ A, np.eye(A.shape[0]))

if __name__ == '__main__':
    A = np.array([[0.33333333, 0.66666667, 0.66666667],
                  [0.66666667, -0.66666667, 0.33333333],
                  [0.66666667, 0.33333333, -0.66666667], ])
    print(is_orthogonal_matrix(A))  # True
