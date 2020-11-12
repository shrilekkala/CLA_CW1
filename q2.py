from numpy import *
from cla_utils import *
import scipy
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

# Construct a Vandermonde matrix so the least squares polynomial curve can be plotted
def Vandermonde(m):

    V = np.ones((100,m))
    for i in range (1,m):
        V[:,i] = x_vals**i
    
    return V

# generate evenly distributed values from -1 to 1
x_vals = np.arange(-1,1.00,0.02)

    
m=11

A,b = linear_system(m)
coef = least_squares(A,b)

# generate the y values to be plotted using the least squares polynomial
y_vals = Vandermonde(m) @ coef

# plot of original data
plt.plot(x,f) 

# plot of the  polynomial resulting from  least squares
plt.plot(x_vals,y_vals) 

plt.show()