"""
This module contains functions for performing Gaussian elimination on a matrix.
It is important to note that gaussian elimination is not the most optimal
solution to solving a system of linear equations. This is because of the
complexity, O(n^3), a lot of storage is required unnecessarily (lots of zeros
are stored on the lower triangular part of the matrix), and the algorithm is
not stable (rounding errors can accumulate).

I have written to code to perform Gaussian elimination on a numpy array, a
torch tensor, and a tensorflow tensor. The specific algorithm comes from some
psuedocode from the textbook. The implementations are straightforward, except
for the tensorflow implementation. tf tensors are immutable, so I had to
convert the tf tensor to a numpy array, perform the elimination, and then
convert the result back to a tf tensor. I am still juggling with the tf design
philosophy and programming model, so I am not sure if this is best or not.
"""

# something tensorflow said
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import torch
import tensorflow as tf
from typing import Optional
from __init__ import generate_square_matrix_np, generate_augmented_matrix_np, generate_square_matrix_torch, generate_augmented_matrix_torch, generate_square_matrix_tf, generate_augmented_matrix_tf

def gaussian_elimination_np(A_orig: np.array) -> Optional[np.array]:
    """
    Perform Gaussian elimination on a matrix A. Can also solve a system of
    linear equations.

    :param A: a matrix (np array)
    :return: the reduced row echelon form of A, or None if A is irregular
    """
    A = A_orig.copy()
    n = len(A)

    for j in range(n):
        if A[j, j] == 0:
            print("A is irregular")
            return None

        else:
            for i in range(j + 1, n):
                factor = A[i, j] / A[j, j]
                A[i, j:] = np.subtract(A[i, j:], factor * A[j, j:])
    return A

def gaussian_elimination_torch(A_orig: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Perform Gaussian elimination on a matrix A. Can also solve a system of
    linear equations.

    :param A: a matrix (torch tensor)
    :return: the reduced row echelon form of A, or None if A is irregular
    """
    A = A_orig.clone()
    n = A.size(0)

    for j in range(n):
        if A[j, j] == 0:
            print("A is irregular")
            return None

        else:
            for i in range(j + 1, n):
                factor = A[i, j] / A[j, j]
                A[i, j:] = A[i, j:] - factor * A[j, j:]
    return A

def gaussian_elimination_tf(A_orig: tf.Tensor) -> Optional[tf.Tensor]:
    """
    Perform Gaussian elimination on a matrix A. Can also solve a system of
    linear equations.

    :param A: a matrix (tf tensor)
    :return: the reduced row echelon form of A, or None if A is irregular
    """
    A = A_orig.numpy()
    numpy_elim = gaussian_elimination_np(A)
    if numpy_elim is None:
        return None

    return tf.convert_to_tensor(numpy_elim)


def back_substitution_np(U_orig: np.array) -> Optional[np.array]:
    """
    Perform back substitution on a matrix A. Can only be performed on upper
    triangular matrices.
    Ux = c

    :param A: an upper triangle matrix, n x n+1 (np array)
    :return: the solution to the system of linear equations, or None if A is
    irregular
    """
    U = U_orig.copy()
    n = len(U)

    x_solutions = np.zeros(n)
    x_solutions[-1] = U[-1, -1] / U[-1, -2]
    # x_n = c_n / U_nn

    for i in range(n-1, -1, -1):
        summation = 0
        for j in range(i+1, n):
            summation += U[i, j] * x_solutions[j]
            # \sum_{j=i+1}^{n} U_ij * x_j

        x_solutions[i] = (U[i, n] - summation) / U[i, i]
        # x_i = 1/u_ii * (c_i - sum)

    return x_solutions



if __name__ == "__main__":
    # torch_A = generate_augmented_matrix_torch(5, dtype="float", seed=42)
    # print(torch_A)
    # print(gaussian_elimination_torch(torch_A))
    #
    # tf_A = generate_augmented_matrix_tf(5, dtype="float", seed=42)
    # print(tf_A)
    # print(gaussian_elimination_tf(tf_A))

    A = np.array([[2, 6, 4, 8, 1], [1, 3, 1, 2, 3], [0, 0, 2, 4, 1]], dtype=float)
    U = gaussian_elimination_np(A)
    solutions = back_substitution_np(U)

    print(U)
    print(solutions)
