from numpy import *
from cla_utils import *
from scipy import linalg_solve_triangular
from matplotlib import pyplot as plt 

x = arange(-1,1.1,0.2)
f = 0.0*x
f[3:6] = 1

# construct a linear system given m
def linear_system(m):

    # construct matrix A
    A = np.ones((11,m))
    for i in range (1,m):
        A[:,i] = x**i

    # construct vector b
    b = f.copy()

    return(A,b)

# Solve the least squares system using the QR method
def least_squares(A,b):

    # obtain Q and R by modified Gram-Schmidt
    Q, R = GS_modified(A)
    
    # solve for x using back substitution
    coefficients = scipy.linalg.solve_triangular(R, Q.T @ b)

    return coefficients

A,b = linear_system(10)
coef = least_squares(A,b)
f_ls = A @ coef

# plot of original data
plt.plot(x,f) 

# plot of the resulting polynomial using least squares
plt.plot(x,f_ls) 
plt.show()