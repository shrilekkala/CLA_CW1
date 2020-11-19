import numpy as np
from cla_utils import *
import scipy
from matplotlib import pyplot as plt 


data = np.loadtxt('pressure.dat')

def Vand(x, p):
    """
    Given a polynomial degree p and an array pf points x,
    construct the Vandermonde matrix V for use in the linear system Vx = b

    :return V: the matrix V in the linear system
    """
    V = np.vander(x, p)
    V = np.flip(V, axis = 1)

    return V

def generate_mats_1(p):
    """
    Given a polynomial degree p
    construct the matrices A, B and the vectors b, d
    for use in the linear system Ax=b with constraint Cy=d 
    as required in question 4b)

    :return A, B, b, d
    """
    # construct A as a block diagonal matrix of dimensions 100 x 2(p+1)
    A = np.zeros((100,2*(p+1)))
    A[:51,:p+1] = Vand(data[:51,0], p+1)
    A[51:,p+1:] = Vand(data[51:,0], p+1)

    # construct B of dimensions 2 x 2(p+1_)
    B = np.zeros((2,2*(p+1)))
    B[0, :p+1] = np.array([1 * data[50,0]**i for i in range(p+1)])
    B[0, p+1:] = - B[0, :p+1]
    B[1, :p+1] = np.multiply(np.array([1 * data[50,0]**(i-1) for i in range(p+1)]), np.arange(p+1))
    B[1, p+1:] = - np.multiply(np.array([1 * data[50,0]**(i-1) for i in range(p+1)]), np.arange(p+1))

    # constrict vector b of dimensions 100 x 1
    b = data[:,1]

    # constrict vector d of dimensions 2 x 1
    d = np.array([0,-5]).T
    return A, B, b, d

def construct_A2(theta_i, M):
    """
    Given the data points theta_i and the number of intervals M
    construct the Vandermonde matrix A
    for use in the linear system  as required in question 4d)

    :return A
    """
    # list storing Vandermonde matrices for each interval
    A_mats = list()

    for m in range(M):
        # obtain the index of theta values that are between 2*m*pi/M and 2*(m+1)*pi/M
        index = np.where(np.logical_and(theta_i>=2*np.pi*m/M, theta_i<2*np.pi*(m+1)/M))

        # construct the Vandermonde matrix for the polynomial in each interval
        A_mats.append(Vand(theta_i[index], p+1))
        

    # construct the block diagonal matrix A
    A = A_mats[0]
    for i in range(1,M):
        A = scipy.linalg.block_diag(A, A_mats[i])

    return A

def construct_B2(M):
    """
    Given the number of intervals M
    construct the matrix B for use in the linear system  as required in question 4d)

    :return B
    """
    B = np.zeros((2*M,M*(p+1)))

    for i in range(M-1):
        # enforce the continuity constraint
        B[i,i*(p+1):(i+1)*(p+1)] = np.array([1 * (2*np.pi*(i+1) / M)**j for j in range(p+1)])
        B[i,(i+1)*(p+1):(i+2)*(p+1)] = - B[i,i*(p+1):(i+1)*(p+1)]
                                            
        # enforce the continuity of derivative constraint
        B[i+M,i*(p+1):(i+1)*(p+1)] = np.multiply(np.array([1 * (2*np.pi*(i+1) / M)**(j-1) for j in range(p+1)]), np.arange(p+1))
        B[i+M,(i+1)*(p+1):(i+2)*(p+1)] = - B[i+M,i*(p+1):(i+1)*(p+1)]

    # edge case for continuity (last interval and first interval)
    B[M-1,(M-1)*(p+1):M*(p+1)] = np.array([1 * (2*np.pi)**j for j in range(p+1)])
    B[M-1, 0] = -1

    # edge case for continuity of derivative (last interval and first interval)
    B[2*M-1,(M-1)*(p+1):M*(p+1)] = np.multiply(np.array([1 * (2*np.pi)**(j-1) for j in range(p+1)]), np.arange(p+1))
    B[2*M-1, 1] = -1

    return B

def generate_mats_2(p, M):
    """
    Given a polynomial degree p
    construct the matrices A, B and the vectors b, d
    for use in the linear system Ax=b with constraint Cy=d 
    as required in question 4d)

    :return A, B, b, d
    """
    A = construct_A2(theta_i, M)
    B = construct_B2(M)
    b = np.concatenate((x_i[:, np.newaxis],y_i[:, np.newaxis]), axis=1)
    d = np.zeros((2*M, 2))

    return A, B, b, d

# function that generates the piecwise least squares polynomial
def piecewise_poly(p, M, q):
    """
    Given a polynomial degree p, an integer M and a question number q
    generate the piecewise least squares polynomial for question 4c) or 4d)
    using the QR method described in the report

    :param p: degree of polynomial approximation
    :param M: number of intervals on which to split the polynomial piecewise
    :param q: question number (1 or 2 referring to 4c) or 4d) respectively)

    :return x: the coefficients of the required piecewise polynomial approximation
    """
    # construct the appropriate matrices for the question
    if q == 1:
        A, B, b, d = generate_mats_1(p)
    elif q == 2:
        A, B, b, d = generate_mats_2(p, M)

    # number of rows of B
    k = np.shape(B)[0]

    # obtain the full QR factorisation of B^T
    Qb, Rb = householder_full_qr(B.T)
    Rb = Rb[:k,:k]

    # solve for y_1 by back substitution
    y1 = scipy.linalg.solve_triangular(Rb.T, d)

    # construct the matrices A_1 and A_2
    A1 = A @ Qb[:, :k]
    A2 = A @ Qb[:, k:]

    # obtain the reduced QR factorisation of A_2
    Qa2, Ra2 = householder_qr(A2)

    # solve the least squares model for y_2
    y2 = scipy.linalg.solve_triangular(Ra2, Qa2.T @ (b - A1 @ y1))

    # substitute y1 and y2 back to get x 
    if q == 1:
        x = Qb @ np.vstack((y1[:, np.newaxis], y2[:, np.newaxis]))
    elif q == 2:
        x = Qb @ np.vstack((y1, y2))

    return x

def ellipse_with_noise(n, a, b):
    """
    Given a factor of perturbation "n", the x radius "a", and the y radius "b",
    generate points of an ellipse in the x-y plane with x radius a and y radius b,
    randomly perturb the x and y values with a small vector from the multivariate normal distribution

    :return theta_i, x_i, y_i
    """
    # generate random thetas from Uniform(0, 2pi) and sort them in increasing order
    theta_i = np.sort(np.random.uniform(0,2*np.pi,(n)))

    # generate an ellipse with random noise using the theta values
    x_i = 5 + a * np.cos(theta_i) + np.random.multivariate_normal(np.zeros(n),np.eye(n)/1)
    y_i = 3 + b * np.sin(theta_i) + np.random.multivariate_normal(np.zeros(n),np.eye(n)/1)
        
    return theta_i, x_i, y_i


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


""" Question 4c) """

# plot the observed data
plt.scatter(data[:,0] ,data[:,1], c="r", marker='x', label = "Observed Data") 
plt.xlabel("Depth")
plt.ylabel("Pressure")

# generate evenly distributed values from 0 to 2
x_vals = np.arange(0,2.02,0.01)

# fix the degree p
p = 10

# obtain the coefficients of least squares solution
x = piecewise_poly(p, 2, 1)

# find the y values for the each piece of the polynomial approximation using x_vals
y_vals = x_vals.copy()[:, np.newaxis]
y_vals[:101] = Vand(x_vals[:101], p+1) @ x[:p+1]
y_vals[101:] = Vand(x_vals[101:], p+1) @ x[p+1:]

# plot the piecewise polynomial approximation
plt.plot(x_vals[:101], y_vals[:101])
plt.plot(x_vals[100:], y_vals[100:])
plt.title("Least squares polynomial of degree " + str(p) +" for the data")

plt.show()

""" Investigating the effect of varying p """

# create an array that stores the Residual Sum of Squares for the least squares solution for different p
RSS_mat = np.zeros(20)

# vary p from 1 to 20 and store the resulting RSS in the array for comparison
for i in range(1,21):
    A = generate_mats_1(i)[0]
    x = piecewise_poly(i, 2, 1)
    RSS = np.sum ((A @ x - data[:,1][:, np.newaxis]) ** 2)
    RSS_mat[i-1] = RSS

# plot the RSS against different values of p
plt.semilogy(range(1,21), RSS_mat)
plt.title("Semilog plot of the RSS against polynomial degree")
plt.xlabel("p (polynomial degree)")
plt.ylabel("RSS")
plt.show()


""" Question 4d) """

# construct a smooth closed curve to be used as an example dataset for question 4 d)
a = 20
b = 10
theta_i, x_i, y_i = ellipse_with_noise(250,a,b)

# plot the curve
plt.scatter(x_i,y_i, c="r", marker='x', label = "Ellipse") 
plt.xlabel("x")
plt.ylabel("y")

# adjust parameter values of M (number of intervals) and p (degree of polynomial)
M = 10
p = 5

# obtain the coefficients of the least squares solution
x = piecewise_poly(p, M, 2)

# generate evenly distributed theta values from 0 to 2 pi
theta_vals=np.linspace(0,2*np.pi,500)

# find the x and y values for the piecwise polynomial approximation using theta_vals
xy_vals = construct_A2(theta_vals, M) @ x

# plot the piecewise polynomial approximation
plt.plot(xy_vals[:,0], xy_vals[:,1])
plt.title("Least squares polynomial of degree " + str(p) +" for the data, with M = " + str(M))

plt.show()

