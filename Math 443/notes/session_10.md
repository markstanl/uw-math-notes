# Session 10

## Linear Independence

Key Idea:

1. A set of vectors are linearly dependent if there is a nonzero solution to $$Ac=0$$
2. A set of vectors are linearly independent if the only solution to $$Ac=0$$ is the trivial solution $$c=0$$
3. A vector $b$ is in the span of a set of vectors if there is a solution to $$Ac=b$$

### Lemma

Any collection of $k > n$ vectors in $\mathbb{R}^n$ is linearly dependent.

#### Justification:

Because there are more columns than rows, the system in it's most reduced form will have a free variable (something
like $c_xx+c_yy=0$), which gives us a free variable and a nontrivial solution.

### Proposition

A set of $k$ vectors in $\mathbb{R}^n$ is linearly independent iff the corresponding $n \times k$ matrix $A$ has rank
$n$.

### Theorem:
A set of $n$ vectors in $\mathbb{R}^n$ is a basis iff the corresponding matrix is nonsingular (full rank $n$).

## Basis
A basis of vector space $V$ is a finite collection of elements$v_1, v_2, \ldots, v_n$ in $V$ such that:
1. Span$(v_1, v_2, \ldots, v_n) = V$
2. The set $(v_1, v_2, \ldots, v_n)$ is linearly independent.