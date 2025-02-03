import numpy as np
from typing import Optional


def gauss_jordan_elimination(A_orig: np.array) -> Optional[np.array]:
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

    return augmented_matrix[:, n:]



if __name__ == '__main__':
    A = np.array([[0, 2, 1], [2, 6, 1], [1, 1, 4]], dtype=float)
    A_inverse = gauss_jordan_elimination(A)
    print(A_inverse)

    print(np.matmul(A, A_inverse))
    
    # A = np.array([[1, -2], [3, -3]])
    # B = np.array([[1, 3], [3, 1]])
    # AB = np.matmul(A, B)
    #
    # A_inverse = gauss_jordan_elimination(A)
    # B_inverse = gauss_jordan_elimination(B)
    # AB_inverse = gauss_jordan_elimination(AB)
    #
    # print(A_inverse)
    # print(B_inverse)
    # print(AB_inverse)
    # print(np.matmul(B_inverse, A_inverse))

