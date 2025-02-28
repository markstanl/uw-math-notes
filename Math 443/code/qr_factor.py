import torch as t

def qr_factor(A: t.Tensor) -> t.Tensor or None:
    """
    Compute the QR factorization of a matrix.

    :param A: the matrix
    :return: the Q and R factors, or None if the matrix has linearly dependent columns
    """
    n = A.shape[0]
    Q = t.zeros_like(A)
    R = t.zeros((n, n))
    A_copy = A.clone()

    for j in range(n):
        R[j, j] = t.linalg.norm(A_copy[:, j])
        if R[j, j] == 0:
            print("Matrix has linearly dependent columns.")
            return None

        Q[:, j] = A_copy[:, j] / R[j, j]

        for k in range(j + 1, n):
            R[j, k] = t.dot(Q[:, j], A_copy[:, k])
            A_copy[:, k] -= R[j, k] * Q[:, j]

    return Q, R

if __name__ == '__main__':
    A = t.tensor([[1., 1., 2., 4.], [1., 0., -2., 6.], [-1., 2., 3., 9.]])
    Q, R = qr_factor(A)
    print("Q:\n", Q)
    print("R:\n", R)
