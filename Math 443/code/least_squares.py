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


if __name__ == '__main__':
    A = np.array([[1, 2], [3, -1], [-1, 2], [1, -1], [2, 1]])
    b = np.array([1, 0, -1, 2, 2])
    x = least_squares(A, b)
    print("Least squares solution:", x)

    e = compute_least_squares_error(A, b, x)
    print("Least squares error:", e)