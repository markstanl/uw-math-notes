import numpy as np

def norm(vector):
    """
    Compute the norm of a vector, the following three ways to get the euclidean
    norm are equivalent.

    1. Compute the square root of the inner product of the vector with itself.
    2. Compute the square root of the sum of the squares of the elements of the vector.
    3. Use the numpy function np.linalg.norm.

    :param vector: the original vector
    :return: the norm
    """
    # inner_product = np.dot(vector, vector)
    # return np.sqrt(inner_product)
    # vector = vector ** 2
    # return np.sqrt(np.sum(vector))
    return np.linalg.norm(vector)


def verify_triangle_inequality(v, w):
    """
    Verify the triangle inequality for two vectors v and w.

    :param v: the first vector
    :param w: the second vector
    :return: True if the triangle inequality holds, False otherwise
    """
    norm_v = norm(v)
    norm_w = norm(w)
    norm_v_plus_w = norm(v + w)
    print(f"||v||={norm_v}, ||w||={norm_w}, ||v||+||w||= {norm_v+norm_w}, ||v + w|| = {norm_v_plus_w}")
    return norm_v_plus_w <= (norm_v + norm_w)

def is_orthogonal(u: np.ndarray, v: np.ndarray) -> bool:
    return np.inner(u, v) == 0

def verify_orthonormal_basis(vectors: np.ndarray) -> bool:
    """
    Verify that a set of vectors forms an orthonormal basis.

    :param vectors: the set of vectors
    :return:
    """
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            if not np.isclose(np.inner(vectors[i], vectors[j]), 0):
                print(f"inner product of {vectors[i]} and {vectors[j]} is {np.inner(vectors[i], vectors[j])}")
                return False

    for vector in vectors:
        if not np.isclose(norm(vector), 1):
            print(f"norm of {vector} is {norm(vector)}")
            return False

    return True

def convert_to_orthonormal_basis(vectors: np.ndarray) -> np.ndarray:
    n = len(vectors)
    u = np.zeros_like(vectors)
    u[0] = vectors[0]
    for i in range(1, n):
        u[i] = vectors[i]
        for j in range(i):
            u[i] -= np.inner(vectors[i], u[j]) / np.linalg.norm(u[j])**2 * u[j]
    print(u)
    return np.array([vector / np.linalg.norm(vector) for vector in u])

if __name__ == '__main__':
    v_1 = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], dtype=float)
    v_2 = np.array([1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=float)
    v_3 = np.array([1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)], dtype=float)
    # yes
    vectors = np.array([v_1, v_2, v_3])

    print(verify_orthonormal_basis(vectors))




