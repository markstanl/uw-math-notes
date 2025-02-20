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

if __name__ == '__main__':
    v = np.array([[1], [2]])
    w = np.array([[-1], [2]])

    inner_product = np.dot(v.T, w)

    norm_v = norm(v)
    norm_w = norm(w)

    # print(f"<v, w> = {inner_product}, ||v|| = {norm_v}, ||w|| = {norm_w}")
    # print(f"||v|| * ||w|| = {norm_v * norm_w}")
    # print(inner_product <= norm_v * norm_w)


    v2 = np.array([[1], [1], [1], [1]])
    w2 = np.array([[1], [1], [1], [-1]])

    inner_product = np.dot(v2.T, v2)

    norm_v = norm(v2)
    norm_w = norm(w2)

    print(f"<v, w> = {inner_product}, ||v|| = {norm_v}, ||w|| = {norm_w}")
    print(f"||v|| * ||w|| = {norm_v * norm_w}")
    print(inner_product <= norm_v * norm_w)

    cos_theta = inner_product / (norm_v * norm_w)
    theta = np.arccos(cos_theta)
    #
    # print(cos_theta)
    # print(theta)


    v = np.array([[1], [2], [-1]])
    w = np.array([[2], [0], [3]])
    # verify_triangle_inequality(v, w)
    v = np.array([[1], [2], [3]])
    w = np.array([[1], [-1], [2]])
    print(verify_triangle_inequality(v, w))
