import numpy as np

def least_squares(A, b):
    """
    Solve the least squares problem min ||Ax - b||_2,
    assuming A is invertible
    """
    return (A.T @ b) @ np.linalg.inv(A.T @ A)

def compute_least_squares_error(A, b, x):
    """
    Compute the least squares error ||Ax - b||_2
    """
    return (np.linalg.norm(A @ x - b)) ** 2


def weighted_least_squares(A: np.ndarray,
                           b: np.ndarray,
                           C: np.ndarray) -> np.ndarray:
    print(A)
    print(b)
    print(C)
    A_transpose_C_A = A.T @ C @ A
    A_transpose_C_b = A.T @ C @ b
    print(A_transpose_C_A)
    print(A_transpose_C_b)
    print(np.concatenate((A_transpose_C_A, A_transpose_C_b[:, None]), axis=1))
    return np.linalg.solve(A_transpose_C_A, A_transpose_C_b)

if __name__ == '__main__':
    A = np.array([[1, 2], [3, -1], [-1, 2], [1, -1], [2, 1]])
    b = np.array([1, 0, -1, 2, 2])
    x = least_squares(A, b)
    print("Least squares solution:", x)

    e = compute_least_squares_error(A, b, x)
    print("Least squares error:", e)

    # weighted least squares
    A = np.array([[1, 0], [1, 1], [1, 3], [1, 6]])
    b = np.array([2, 3, 7, 12])
    C = np.diag([3, 2, 1, 1])
    x = weighted_least_squares(A, b, C)
    print(x)

    print(np.linalg.solve(np.array([[2, 5], [10, 46]]), np.array([12, 96])))
