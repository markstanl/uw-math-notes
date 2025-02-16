import numpy as np

def rre(A_orig: np.array) -> np.array or None:
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
    Computes the rank of a matrix A. Does so by using row reduction to
    get the reduced matrix, and then counting the number of non-zero rows.

    :param A: the matrix A
    :return: the rank of A
    """
    A_rre = rre(A)
    return len([row for row in A_rre if any(row)])

if __name__ == '__main__':
    # Problem 2
    A = np.array([[2, 6, 4, 8], [1, 3, 1, 2], [0, 0, 2, 4]], dtype=float)
    A_rre = rre(A)
    print(A_rre)

    # Problem 3
    # part a
    A = np.array([[2, -2, 4, 0], [1, -1, 2, 0], [3, -3, 7, 0]], dtype=float)
    print(A)
    A_rre = rre(A)
    print(A_rre)

    # part d
    print('part d')
    b = np.array([[1], [2], [4]], dtype=float)
    augmented_matrix = np.concatenate((A, b), axis=1)
    print(augmented_matrix)
    A_rre = rre(augmented_matrix)
    print(A_rre)

    # Problem 4
    A = np.array([[0, 1, 2, 2], [0, 3, 8, 7], [0, 0, 4, 2]], dtype=float)
    b = np.array([[0], [0], [0]], dtype=float)
    augmented_matrix = np.concatenate((A, b), axis=1)
    augmented_matrix_rre = rre(augmented_matrix)
    print(augmented_matrix_rre)
    print(rank(A))

    # part c
    B_row = np.concatenate((A, A), axis=1)
    B = np.concatenate((B_row, B_row), axis=0)
    print(B.shape)
    print(B)

    print(rre(B))
    print(rank(B))

