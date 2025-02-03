import numpy as np
import tensorflow as tf
import torch

def generate_square_matrix_np(n, dtype="float"):
    """
    Generate a square matrix of size n x n with random entries using numpy
    arrays.
    """
    if dtype == "int":
        return np.array([[np.random.randint(5) for _ in range(n)] for _ in range(n)], dtype=int)
    elif dtype == "float":
        return np.array([[np.random.rand() for _ in range(n)] for _ in range(n)], dtype=float)
    else:
        raise ValueError("dtype must be either 'int' or 'float'")

def generate_augmented_matrix_np(n, dtype="float"):
    """
    Generate an augmented matrix of size n x (n + 1) with random entries using
    numpy arrays.
    """
    if dtype == "int":
        return np.array([[np.random.randint(5) for _ in range(n + 1)] for _ in range(n)], dtype=int)
    elif dtype == "float":
        return np.array([[np.random.rand() for _ in range(n + 1)] for _ in range(n)], dtype=float)
    else:
        raise ValueError("dtype must be either 'int' or 'float'")

def generate_square_matrix_torch(n, dtype="float", seed=None):
    """
    Generate a square matrix of size n x n with random integers using torch
    tensors.
    """
    if seed:
        torch.manual_seed(seed)

    if dtype == "int":
        return torch.randint(5, (n, n))
    elif dtype == "float":
        return torch.rand(n, n)

    else:
        raise ValueError("dtype must be either 'int' or 'float'")

def generate_augmented_matrix_torch(n, dtype="float", seed=None):
    """
    Generate an augmented matrix of size n x (n + 1) with random entries using
    torch tensors.
    """
    if seed:
        torch.manual_seed(seed)

    if dtype == "int":
        return torch.randint(5, (n, n + 1))
    elif dtype == "float":
        return torch.rand(n, n + 1)
    else:
        raise ValueError("dtype must be either 'int' or 'float'")


def generate_square_matrix_tf(n, dtype="float", seed=None):
    """
    Generate a square matrix of size n x n with random integers using tensorflow
    tensors.
    """
    if seed:
        generator = tf.random.Generator.from_seed(seed)
    else:
        generator = tf.random.Generator.from_non_deterministic_state()

    if dtype == "int":
        return generator.uniform((n, n), maxval=5, dtype=tf.int32)

    else:
        return generator.uniform((n, n), dtype=tf.float32)

def generate_augmented_matrix_tf(n, dtype="float", seed=None):
    """
    Generate an augmented matrix of size n x (n + 1) with random entries using
    tensorflow tensors.
    """
    if seed:
        generator = tf.random.Generator.from_seed(seed)
    else:
        generator = tf.random.Generator.from_non_deterministic_state()

    if dtype == "int":
        return generator.uniform((n, n + 1), maxval=5, dtype=tf.int32)

    else:
        return generator.uniform((n, n + 1), dtype=tf.float32)




if __name__ == "__main__":
    print(generate_square_matrix_torch(5, dtype="int", seed=42))