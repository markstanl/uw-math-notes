# Quadratic Minimization
Quadratic functions are pretty straigthfoward, functions of degree 2

# Multivariate Case
$$p(x)=x^TKx-2x^Tf+x, \quad x \in \mathbb{R}^n$$
Where:
- $$K$$ is a symmetric matrix
- $$f$$ is a constant vector
- $$x$$ is the constant scalar

### Theorem
If $$K$$ is positive definite, then the minimizer of $$p(x)$$ is the solution to the system of equations:
$$x^*=K^{-1}f$$
Minimum value
$$p(x^*)=c-f^TK^{-1}f$$