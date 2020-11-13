from numpy import *
from cla_utils import *
import scipy
from matplotlib import pyplot as plt 

x = arange(-1,1.1,0.2)
f = 0.0*x
f[3:6] = 1

# construct the linear system given m, x
def linear_system(m, x):

    # construct matrix A
    A = np.ones((11,m+1))
    for i in range (1,m+1):
        A[:,i] = x**i

    return(A)

# Solve the least squares system using the QR method
def least_squares(A,b):

    # obtain Q and R by modified Gram-Schmidt
    Q, R = GS_modified(A)
    
    # solve for x using back substitution
    coefficients = scipy.linalg.solve_triangular(R, Q.T @ b)

    return coefficients

# Construct a Vandermonde matrix so the least squares polynomial curve can be plotted
def Vandermonde(m):

    V = np.ones((101,m+1))
    for i in range (1,m+1):
        V[:,i] = x_vals ** i
    
    return V

# generate evenly distributed values from -1 to 1
x_vals = np.arange(-1,1.02,0.02)

# function generates the plots of the least squares polynomial for the data given an m, x and f
def plot_ls_poly(m, x, f):

    A = linear_system(m,x)
    coef = least_squares(A,f)

    # generate the y values to be plotted using the least squares polynomial
    y_vals = Vandermonde(m) @ coef

    # plot of the  polynomial resulting from  least squares
    plt.plot(x_vals,y_vals, zorder=0, label = "Least Squares model")

# generate a small random vector of the same length as f using the multivariate normal distribution
def pert(f):

    df = np.random.multivariate_normal(np.zeros(len(f)),eye(len(f))/100)
    return df



plot_ls_poly(10, x, f)
plt.title("Least squares polynomial of degree 10 for the data")
plt.scatter(x,f, c="r", marker='x', label = "Observed Data") 
plt.legend(loc='best')
plt.show()

plot_ls_poly(7, x, f)
plt.scatter(x,f, c="r", marker='x', label = "Observed Data") 
plt.title("Least squares polynomial of degree 7 for the data")
plt.legend(loc='best')
plt.show()

# Investigate the sensitivity of the 2 polynomials above under the small random perturbations
# for a given m (degree of poly) and n (number of random perturbations to data)
def pert_sensitivity(m, n):

    A = linear_system(m,x)
    coef = least_squares(A,f)
    SSD = np.zeros(n)
    for i in range(n):
        dcoef = least_squares(A, f+pert(f))
        # calculate the sum of squared differences between coef and dcoef
        SSD[i] = np.linalg.norm (coef - dcoef)
    return SSD


plt.title("Least squares polynomials of degree 10 for the randomly perturbed data")
plt.scatter(x,f, c="r", marker='x', label = "Observed Data") 
for i in range(10):
    plot_ls_poly(10, x, f + pert(f))
plt.show()

plt.title("Least squares polynomials of degree 7 for the randomly perturbed data")
plt.scatter(x,f, c="r", marker='x', label = "Observed Data") 
for i in range(10):
    plot_ls_poly(7, x, f + pert(f))
plt.show()

print(pert_sensitivity(10, 15))
print(pert_sensitivity(7, 15))
