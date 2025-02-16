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

    return augmented_matrix

def rre(A_orig: np.array) -> Optional[np.array]:
    """
    Performs row reduction to get a matrix into reduced row echelon form.
    Pseudocode from https://rosettacode.org/wiki/Reduced_row_echelon_form

    :param A_orig: the original square matrix A
    :return: the rre form of matrix A
    """
    A = A_orig.copy()
    lead = 0
    rowCount = len(A)
    columnCount = len(A[0])

    for r in range(rowCount):
        if lead >= columnCount:
            return A

        i = r
        while A[i][lead] == 0:
            i += 1
            if i == rowCount:
                i = r
                lead += 1
                if columnCount == lead:
                    return A
        A[[i, r]] = A[[r, i]]

        if A[r][lead] != 0:
            A[r] = A[r] / A[r][lead]

        for i in range(rowCount):
            if i != r:
                A[i] = A[i] - A[i][lead] * A[r]
        lead += 1

    return A

def rank(A: np.array) -> int:
    """
    Computes the rank of a matrix A. Does so by using gaussian elimination to
    get the reduced matrix, and then counting the number of non-zero rows.

    :param A: the matrix A
    :return: the rank of A
    """
    A_rre = rre(A)
    return len([row for row in A_rre if any(row)])

if __name__ == '__main__':
    # A = np.array([[1, 2, -1, -4], [2, 3, -1, -11], [-2, 0, -3, 22]], dtype=float)
    # B = np.array([[2, 4, -2], [-4, -7, 4], [6, 8, -6]], dtype=float)
    # B_rre = rre(B)
    # print(B_rre)

    A = np.array([[2, -2, 4, 0], [1, -1, 2, 0], [3, -3, 7, 0]], dtype=float)
    A = A.T
    print(A)
    A_rre = rre(A)
    print(A_rre)

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

