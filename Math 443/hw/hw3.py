import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('TkAgg')


def problem_3():
    def projection(a: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.dot(v, a) / np.dot(a, a) * a
    a = np.array([2, 1, 3])
    v = np.array([1, 1, 1])

    P = np.array([[2/7, 1/7, 3/7], [1/7, 1/14, 3/14], [3/7, 3/14, 9/14]])
    print(np.linalg.matrix_rank(P))

def problem_3b_col():
    a = np.array([2, 1, 3])
    P = np.array([[2/7, 1/7, 3/7], [1/7, 1/14, 3/14], [3/7, 3/14, 9/14]])
    c1, c2, c3 = P[:, 0], P[:, 1], P[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    origin = np.array([0, 0, 0])

    ax.quiver(*origin, *c1, color='b', arrow_length_ratio=0.1)
    ax.quiver(*origin, *c2, color='r', arrow_length_ratio=0.1)
    ax.quiver(*origin, *c3, color='g', arrow_length_ratio=0.1)
    ax.quiver(*origin, *a, color='y', arrow_length_ratio=0.3)

    ax.set_xlim([0, 3])
    ax.set_ylim([0, 3])
    ax.set_zlim([0, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def problem_3b_null():
    a = np.array([2, 1, 3])
    P = np.array([[2/7, 1/7, 3/7], [1/7, 1/14, 3/14], [3/7, 3/14, 9/14]])
    origin = np.array([0, 0, 0])

    t = np.linspace(-10, 10, 100)
    x = 2 * t
    y = t
    z = np.zeros_like(t)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z)
    ax.quiver(*origin, *a, color='y', arrow_length_ratio=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 3])
    ax.set_zlim([0, 3])
    ax.legend()

    plt.show()

def inverse(A: np.ndarray) -> np.ndarray:
    """
    The common algorithm for finding the inverse of a matrix

    :param A: The matrix to find the inverse of
    :return: the inverse of A
    """
    A_augmented = np.concatenate((A, np.eye(A.shape[0])), axis=1)
    A_rref = sp.Matrix(A_augmented).rref()
    return np.array(A_rref[0][:, A.shape[0]:])

def problem_4():
    def A_n(n: int) -> np.ndarray:
        A = np.eye(n) - np.ones((n, n))
        return A

    A_4 = A_n(4)
    A_4_inv = inverse(A_4)

    det_A_4 = np.linalg.det(A_4)
    det_A_4_11 = np.linalg.det(A_4[1:, 1:])
    print(A_4_inv[0, 0], det_A_4 / det_A_4_11)


if __name__ == '__main__':
    problem_4()

