import numpy as np


def is_perpendicular(a, b):
    return np.isclose(a @ b, 0)


def verify_orthonormal_basis(vectors: np.ndarray) -> bool:
    """
    Verify that a set of vectors forms an orthonormal basis.

    :param vectors: the set of vectors
    :return: a boolean value for if the vectors form an orthonormal basis
    """
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if not np.isclose(np.inner(vectors[i], vectors[j]), 0):
                print(
                    f"inner product of {vectors[i]} and {vectors[j]} is {np.inner(vectors[i], vectors[j])}")
                return False

    for vector in vectors:
        if not np.isclose(np.linalg.norm(vector), 1):
            print(f"norm of {vector} is {np.linalg.norm(vector)}")
            return False

    return True


def gram_schmidt_process(vectors: np.ndarray) -> np.ndarray:
    n = len(vectors)
    u = np.zeros_like(vectors)
    u[0] = vectors[0]
    for i in range(1, n):
        u[i] = vectors[i]
        for j in range(i):
            u[i] -= np.inner(vectors[i], u[j]) / np.linalg.norm(u[j]) ** 2 * u[
                j]
    return np.array([vector / np.linalg.norm(vector) for vector in u])


def problem1():
    b = np.array([0, 8, 8, 20])
    a = np.array([1, 1, 1, 1])

    x_hat = (a.T @ b) / (a.T @ a)
    print(f"x_hat = {x_hat}")

    p = x_hat * a
    print(f"p = {p}")

    error = b - p
    print(is_perpendicular(a, error))

    norm_error = np.linalg.norm(error)
    print(f"||e|| = {norm_error}")


def problem3():
    v_1 = np.array([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
                   dtype=float)
    v_2 = np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0], dtype=float)
    v_3 = np.array([1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6)],
                   dtype=float)
    vectors = np.array([v_1, v_2, v_3])
    print(verify_orthonormal_basis(vectors))

def problem4():
    a = np.array([1, -1, 0, 0], dtype=float)
    b = np.array([0, 1, -1, 0], dtype=float)
    c = np.array([0, 0, 1, -1], dtype=float)

    orthogonal_basis = gram_schmidt_process(np.array([a, b, c]))
    print(orthogonal_basis)
    print(verify_orthonormal_basis(orthogonal_basis))


if __name__ == "__main__":
    problem4()
