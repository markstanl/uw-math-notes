# Minimization
Consider a function $$f(x,y)=3x^2-2xy+4y^2+x-2y+1$$. Minimization is interested in finding the $$(x*, y*)$$ that 
minimizes the function.

## Minimization for Solving Systems of Equations
Consider the system of equations:
$$f_1(x)=0, \quad f_2(x)=0, \, \ldots \c , f_m(x)=0$$
The solution to this system is the minimizer of the function:
$$p(x)=f_1(x)^2+f_2(x)^2+\ldots+f_m(x)^=\|f(x)\|^2$$

## Linear Systems and Least Squares Problems
A linear system $$Ax=b$$. The solution can be obtained by minimizing the function:
$$p(x)=\|Ax=b\|^2$$

### Least Squares Solution
If the linear system does NOT have a solution. We want to minimize the residual:
$$r = b - Ax$$. We want to minimize the residual:
$$\|r\|=\|Ax-b\|$$
The least squares solution is just the solution $$x^*$$ that minimizes the residual, or solution to the minimization problem.

If the number of layers in a NN is greater than 4, the NN is considered a deep NN.