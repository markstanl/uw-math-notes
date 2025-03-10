import numpy as np


def solve_linear(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(A, b)

def


if __name__ == '__main__':
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    b = np.array([1, 2, 3])

    x = solve_linear(A, b)
    print(x)