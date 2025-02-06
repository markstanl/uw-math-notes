import numpy as np

def gaussian_elim(A_orig: np.array) -> np.array or None:
    """
    Performs gauss jordan elimination on a square matrix to find its inverse.

    :param A_orig: the original square matrix A
    :return: the inverse of A, or None if A is nonsingular
    """
    A = A_orig.copy()
    n = len(A)

    for j in range(n):
        if all([A[k, j] == 0 for k in range(j, n)]):
            print("A is singular")
            return None

        if A[j, j] == 0:
            for k in range(j + 1, n):
                if A[k, j] != 0:
                    A[[j, k]] = A[[k, j]]
                    break

        for i in range(j + 1, n):
            factor = A[i, j] / A[j, j]
            A[i, j:] = np.subtract(A[i, j:], factor * A[j, j:])

    return A

def determinant(A_orig: np.array) -> float or None:
    """
    Computes the determinant of a matrix A
    :param A_orig: the matrix A
    :return: the determinant of A, or None if the matrix is not square
    """
    if len(A_orig) != len(A_orig[0]):
        return None

    det = 1

    A = A_orig.copy()
    n = len(A)

    for j in range(n):
        if all([A[k, j] == 0 for k in range(j, n)]):
            return 0

        if A[j, j] == 0:
            for k in range(j + 1, n):
                if A[k, j] != 0:
                    A[[j, k]] = A[[k, j]]
                    det *= -1
                    break

        for i in range(j + 1, n):
            factor = A[i, j] / A[j, j]
            A[i, j:] = np.subtract(A[i, j:], factor * A[j, j:])
    U = A

    for i in range(n):
        det *= U[i, i]

    return det

def is_invertible(A_orig: np.array) -> bool:
    """
    Checks if a matrix A is invertible.
    :param A_orig: the matrix A
    :return: True if A is invertible, False otherwise
    """
    return determinant(A_orig) != 0

def inverse(A_orig: np.array) -> np.array or None:
    """
    Performs gauss jordan elimination on a square matrix to find its inverse.

    :param A_orig: the original square matrix A
    :return: the inverse of A, or None if A is nonsingular
    """
    A = A_orig.copy()
    n = len(A)
    I = np.identity(n)

    augmented_matrix = np.concatenate((A, I), axis=1)

    for j in range(n):
        if all([augmented_matrix[k, j] == 0 for k in range(j, n)]):
            print("A is singular")
            return None

        if augmented_matrix[j, j] == 0:
            for k in range(j + 1, n):
                if augmented_matrix[k, j] != 0:
                    augmented_matrix[[j, k]] = augmented_matrix[[k, j]]
                    break

        for i in range(j + 1, n):
            factor = augmented_matrix[i, j] / augmented_matrix[j, j]
            augmented_matrix[i, j:] = np.subtract(augmented_matrix[i, j:], factor * augmented_matrix[j, j:])
    # converts augmented_matrix into an upper triangular matrix

    for j in range(n - 1, -1, -1):
        augmented_matrix[j] /= augmented_matrix[j, j]
        for i in range(j):
            factor = augmented_matrix[i, j] / augmented_matrix[j, j]
            augmented_matrix[i, j:] = np.subtract(augmented_matrix[i, j:], factor * augmented_matrix[j, j:])

    return augmented_matrix

if __name__ == '__main__':
    A = np.array([[1, 0, -1, 2], [2, 1, -3, 4], [0, 2, -2, 3], [1, 1, -4, -2]])
    det = determinant(A)
    print(det)
    print(is_invertible(A))