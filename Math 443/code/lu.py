# something tensorflow said
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from typing import Optional, Tuple


def lu_factorization_np(A_orig: np.array) -> Optional[Tuple[np.array, np.array]]:
    """
    Performs LU factorization on a square matrix A. Specifically finds L and U
    through Gaussian elimination, which means that it isn't necessarily quicker
    on a single system of equations than Gaussian elimination.

    :param A: a system of equations, a matrix n x n+1 (np array)
    :return: a tuple containing the upper and lower triangular matrices L and U
    that satisfy A = LU
    """
    A = A_orig.copy()
    n = len(A)

    elementary_matrices = []
    for j in range(n):
        if A[j, j] == 0:
            print("A is irregular")
            return None

        else:
            for i in range(j + 1, n):
                factor = A[i, j] / A[j, j]
                A[i, j:] = np.subtract(A[i, j:], factor * A[j, j:])

                elementary_matrix = np.identity(n)
                elementary_matrix[i, j] = factor
                elementary_matrices.append(elementary_matrix)

    L = elementary_matrices.pop()
    for matrix in reversed(elementary_matrices):
        L = np.matmul(matrix, L)

    return L, A

def back_substitution_np(U_orig: np.array, c_orig: np.array) -> Optional[np.array]:
    """
    Perform back substitution on a matrix U and a vector c to solve the
    system of equations Ux = c.

    :param U_orig: the original square matrix U
    :param c_orig: the original vector c
    :return: the solution vector x
    """
    U = U_orig.copy()
    c = c_orig.copy()
    n = len(U)

    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (c[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        # we can use the dot product as opposed to the summation

    return x

def forward_substitution_np(L_orig: np.array, b_orig: np.array) -> Optional[np.array]:
    """
    Perform forward substitution on a matrix L and a vector b to solve the
    system of equations Lc = b.

    :param L_orig: the original square matrix L
    :param b_orig: the original vector b
    :return: the solution vector c
    """
    L = L_orig.copy()
    b = b_orig.copy()
    n = len(L)

    c = np.zeros(n)
    for i in range(n):
        c[i] = b[i] - np.dot(L[i, :i], c[:i])
        # again we can use the dot product

    return c

def ldv_factorization_np(A_orig: np.array) -> Optional[Tuple[np.array, np.array, np.array]]:
    """
    Performs LU factorization on a square matrix A. Specifically finds L and U
    through Gaussian elimination, which means that it isn't necessarily quicker
    on a single system of equations than Gaussian elimination.

    :param A: a system of equations, a matrix n x n+1 (np array)
    :return: A tuple containing the lower unitriangular matrix L, the diagonal
    matrix D, the upper unitriangular matrix V
    """
    A = A_orig.copy()
    n = len(A)

    elementary_matrices = []
    for j in range(n):
        if all([A[k, j] == 0 for k in range(j, n)]):
            print("A is singular")
            return None

        if A[j, j] == 0:
            for k in range(j + 1, n):
                if A[k, j] != 0:
                    A[[j, k]] = A[[k, j]]
                    break

        else:
            for i in range(j + 1, n):
                factor = A[i, j] / A[j, j]
                A[i, j:] = np.subtract(A[i, j:], factor * A[j, j:])

                elementary_matrix = np.identity(n)
                elementary_matrix[i, j] = factor
                elementary_matrices.append(elementary_matrix)
        print(A)

    print(elementary_matrices)

    L = elementary_matrices.pop()
    for matrix in reversed(elementary_matrices):
        L = np.matmul(matrix, L)

    U = A
    V = A
    D = np.identity(n)
    for j in range(n):
        D[j, j] = U[j, j]
        V[j, j:] = V[j, j:] / V[j, j]

    return L, D, V

def transpose(A: np.array) -> np.array:
    """
    Transpose a matrix A.

    :param A: the matrix to transpose
    """
    n = len(A)
    m = len(A[0])
    A_transpose = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            A_transpose[i, j] = A[j, i]

    return A_transpose

def is_symmetric(A: np.array) -> bool:
    """
    Check if a matrix A is symmetric.

    :param A: the matrix to check
    """
    A_transpose = np.transpose(A)
    return np.array_equal(A_transpose, A)

if __name__ == "__main__":
    # A = np.array([
    #     [4, -2, 1, 0, 0, 0, 0, 0],
    #     [-2, 4, -2, 1, 0, 0, 0, 0],
    #     [1, -2, 4, -2, 1, 0, 0, 0],
    #     [0, 1, -2, 4, -2, 1, 0, 0],
    #     [0, 0, 1, -2, 4, -2, 1, 0],
    #     [0, 0, 0, 1, -2, 4, -2, 1],
    #     [0, 0, 0, 0, 1, -2, 4, -2],
    #     [0, 0, 0, 0, 0, 1, -2, 4]
    # ], dtype=float)
    # b = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    # L, U = lu_factorization_np(A)
    # print(np.matmul(L, U))
    #
    # c = forward_substitution_np(L, b)
    # x = back_substitution_np(U, c)
    #
    # print(np.matmul(A, x))
    # print(b)
    A = np.array([
        [2, 1, 1],
        [4, 5, 2],
        [2, -2, 0]
    ], dtype=float)

    A_2 = np.array([
        [2, 1],
        [4, 3],
    ], dtype=float)

    C = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

    E = np.array([[1, 2, 1], [2, 6, 1], [1, 1, 4]], dtype=float)
    L, D, V = ldv_factorization_np(E)
    print(L)
    print(D)
    print(V)
    print()

    print(L @ D @ V)

