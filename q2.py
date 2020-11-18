import numpy as np
from cla_utils import *
import scipy
from matplotlib import pyplot as plt 

x = np.arange(-1,1.1,0.2)
f = 0.0*x
f[3:6] = 1

def Vandermonde(m, x):
    """
    Given a polynomial degree m and an array pf points x,
    construct the Vandermonde matrix V for use in the linear system Vx = b

    :return V: the matrix V in the linear system
    """
    # construct Vandermonde Matrix A
    V = np.ones((x.shape[0],m+1))
    for i in range (1,m+1):
        V[:,i] = x**i

    return(V)

def least_squares(A,b):
    """
    Given a matrix A and a vector b,
    solve the least squares system to Ax = b using the QR method
 
    :return x: the least squares solution to Ax = b
    """

    # obtain Q and R by modified Gram-Schmidt
    Q, R = GS_modified(A)
    
    # solve for x using back substitution
    coefficients = scipy.linalg.solve_triangular(R, Q.T @ b)

    return coefficients

def plot_ls_poly(m, x, f):
    """
    Given a polynomial degree m, array of points fxand a data f,
    generate the plots of the least squares polynomial for Ax = f
 
    """

    A = Vandermonde(m,x)
    coef = least_squares(A,f)

    # generate the y values to be plotted using the least squares polynomial
    y_vals = Vandermonde(m, x_vals) @ coef

    # plot of the  polynomial resulting from  least squares
    plt.plot(x_vals,y_vals, zorder=0, label = "Least Squares model")

def pert(f):
    """
    Given a vector f,
    perturb f by a small random vector obtained from a multivariate normal distribution
 
    :return df: the randomly perturbed vector
    """
    df = np.random.multivariate_normal(np.zeros(len(f)),np.eye(len(f))/100)
    return df

# Investigate the sensitivity of the 2 polynomials above under the small random perturbations
def pert_sensitivity(m, n):
    """
    Given a polynomial degree m, and a number of times to run n
    Obtain the coefficients vector to Ax = f using least squares.
    Perturb f to obtain df and obtain the coefficents vector to Ax = df using least squares.
    Compute the sum of squared differences between the two coefficients vectors.
    Repeat this process with a different ranodm perturbation n times.
 
    :return SSD: the array of sum of squared differences for each of the n perturbations
    """
    A = Vandermonde(m,x)
    coef = least_squares(A,f)
    SSD = np.zeros(n)
    for i in range(n):
        dcoef = least_squares(A, f+pert(f))
        # calculate the sum of squared differences between coef and dcoef
        SSD[i] = np.linalg.norm (coef - dcoef)
    return SSD

# generate evenly distributed values from -1 to 1 to be used for plotting
x_vals = np.arange(-1,1.02,0.02)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""" Question 2c) """

# plot the original data
plt.scatter(x,f, c="r", marker='x', label = "Observed Data") 

# plot the least squares solution where m = 10
plot_ls_poly(10, x, f)
plt.title("Least squares polynomial of degree 10 for the data")
plt.legend(loc='best')
plt.show()

""" Question 2d) """

plt.title("Least squares polynomials of degree 10 for the randomly perturbed data")

# plot the original data
plt.scatter(x,f, c="r", marker='x', label = "Observed Data") 

# plot the least squares solution where for each random perturbation
for i in range(10):
    plot_ls_poly(10, x, f + pert(f))
plt.show()

# print the sum of squared differences for each perturbation
print("(p = 10) The array of SSDs for each perturbation is:")
print(pert_sensitivity(10, 15))

""" Question 2e) """

# plot the original data
plt.scatter(x,f, c="r", marker='x', label = "Observed Data") 

# plot the least squares solution where m = 7
plot_ls_poly(7, x, f)
plt.title("Least squares polynomial of degree 7 for the data")
plt.legend(loc='best')
plt.show()

"""  Question 2f) """

plt.title("Least squares polynomials of degree 7 for the randomly perturbed data")

# plot the original data
plt.scatter(x,f, c="r", marker='x', label = "Observed Data") 

# plot the least squares solution where for each random perturbation
for i in range(10):
    plot_ls_poly(7, x, f + pert(f))
plt.show()

# print the sum of squared differences for each perturbation
print("(p = 7) The array of SSDs for each perturbation is:")
print(pert_sensitivity(7, 15))
