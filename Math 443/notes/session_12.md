# Review
1. Dot Product: $$\langle v, w \rangle = \sum v_i w_i$$
2. Weighted Inner Product: $$\langle v, w \rangle = \sum c_i v_i w_i$$ where $c_i$ is a constant.

## Dot Product
1. **Definition**: $$v * w = \sum v_i w_i$$
2. **Properties**:
    1. Commutative: $v * w = w * v$
   2. Distributive: $v * (w + u) = v * w + v * u$
   3. Scalar Multiplication: $(cv) * w = c(v * w)$
   4. Relationship to Angle: $$v * w = |v||w|cos(\theta)$$

## Norm
The norm of a vector $v$ is defined as: $$||v|| = \sqrt{\langle v, v \rangle}$$

For continuous functions $f, g$ on $[a, b]$, the inner product is defined as: $$\langle f, g \rangle = \int_a^b f(x)g(x)dx$$

## Cauchy-Schwarz Inequality
For any vectors $v, w$, we have: $$|\langle v, w \rangle| \leq ||v|| ||w||$$