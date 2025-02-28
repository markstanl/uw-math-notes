# Session 15
## Inner Product
$$<u, v>=\sum^n_{i=1}u_iv_i$$
The inner product can be computed in python via the following code:
```python
import numpy as np
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

inner_product = np.inner(u, v)
```
It is noteworthy that this is equivalent to the dot product of the two vectors, and the
matrix multiplication of the transpose of the first vector and the second vector. Thus

```python
import numpy as np
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

inner_product_1 = np.inner(u, v)
inner_product_2 = np.dot(u, v)
inner_product_3 = u.T @ v
# all three are equal
```

## Norm
The norm of a vector is defined as:
$$||u||=\sqrt{\langle u, u \rangle}$$  
In python
```python
import numpy as np
u = np.array([1, 2, 3])

norm = np.linalg.norm(u)
```
The p-norm of a vector is defined as:
$$||u||_p=(\sum^n_{i=1}|u_i|^p)^{1/p}$$  
In python
```python
import numpy as np
u = np.array([1, 2, 3])

p_norm = np.linalg.norm(u, ord=2)
```

## Positive Definite Matrices
A matrix $$A$$ is symmetric iff $$A^T=A$$.
Let $$A$$ be a symmetric matrix of size $$n\times n$$, and take some vector $$v$$ size $$n \times 1$$.
Then, $$A$$ is a positive definite matrix if:  
$$v^TAv>0, \quad \forall x \neq 0$$  
This is equivalent to saying that all the eigenvalues of $$A$$ are positive.

```python
import numpy as np

def is_positive_definie(A: np.ndarray) -> bool:
    return np.all(np.linalg.eigvals(A) > 0)
```

## Orthogonal Vectors
Two vectors $$u$$ and $$v$$ are orthogonal if their inner product is zero.
```python
import numpy as np

def is_orthogonal(u: np.ndarray, v: np.ndarray) -> bool:
    return np.inner(u, v) == 0
```

An orthonormal basis is a set of vectors that are orthogonal and have a norm of 1.

## Linearly Independent to Orthonormal Basis
1. Start with a basis $$\{v_1, v_2, \ldots, v_n\}$$
2. Compute the new orthogonal vectors:
   i. $$u_1 = v_1$$, $$u_2 = v_2 - \frac{<v_2, u_1>}{||u_1||^2}u_1$$, $$u_3 = v_3 - \frac{<v_3, u_1>}{||u_1||^2}u_1 - \frac{<v_3, u_2>}{||u_2||^2}u_2$$, and so on.
3. Normalize the vectors: $$e_i = \frac{u_i}{||u_i||}$$

*_important_*: This will be on the exam.
```python
import numpy as np

def convert_to_orthonormal_basis(vectors: np.ndarray) -> np.ndarray:
    n = len(vectors)
    u = np.zeros_like(vectors)
    u[0] = vectors[0]
    for i in range(1, n):
        u[i] = vectors[i]
        for j in range(i):
            u[i] -= np.inner(vectors[i], u[j]) / np.linalg.norm(u[j])**2 * u[j]
    return np.array([vector / np.linalg.norm(vector) for vector in u])
```

## Vector Projections
The projection of a vector $$v$$ onto a vector $$e$$ is given by:  
$$proj_e(v) = \frac{<v, e>}{<e, e>}e$$
The idea is that it takes the direction of $$e$$ and scales it by the magnitude of $$v$$ in that direction.

## Orthogonal Matrices
A matrix $$A$$ is orthogonal if $$A^TA=I$$, or equivalently, if $$A^T=A^{-1}$$. This
is the same as saying that the columns of $$A$$ form an orthonormal basis.
### Key Properties
1. $$A^T=A^{-1}$$
2. $$||Av|| = ||v||$$
3. $$det(A)= \pm 1$$
