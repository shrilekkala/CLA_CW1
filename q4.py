import numpy as np
from cla_utils import *
import scipy
from matplotlib import pyplot as plt 


data = np.loadtxt('pressure.dat')

# function that generates the required Vandermonde matrix given some data p
def Vand(x, p):

    V = np.vander(x, p)
    V = np.flip(V, axis = 1)

    return V

def generate_mats_1(p):
    # formulate matrices A, B and vectors b, d from the data
    A = np.zeros((100,2*(p+1)))
    A[:51,:p+1] = Vand(data[:51,0], p+1)
    A[51:,p+1:] = Vand(data[51:,0], p+1)

    B = np.zeros((2,2*(p+1)))
    B[0, :p+1] = np.array([1 * data[50,0]**i for i in range(p+1)])
    B[0, p+1:] = - B[0, :p+1]
    B[1, :p+1] = np.multiply(np.array([1 * data[50,0]**(i-1) for i in range(p+1)]), np.arange(p+1))
    B[1, p+1:] = - np.multiply(np.array([1 * data[50,0]**(i-1) for i in range(p+1)]), np.arange(p+1))

    b = data[:,1]

    d = np.array([0,-5]).T
    return A, B, b, d


# function that generates the piecwise least squares polynomial
def piecewise_poly(p,q):
    if q == 1:
        A, B, b, d = generate_mats_1(p)
    else:
        A, B, b, d = generate_mats_2(p)

    # number of rows of B (so we can extend this function for use in part d)
    k = np.shape(B)[0]

    # obtain the full QR of B^T
    Qb, Rb = householder_full_qr(B.T)
    Rb = Rb[:k,:k]

    # solve for y_1 by back substitution
    y1 = scipy.linalg.solve_triangular(Rb.T, d)

    A1 = A @ Qb[:, :k]
    A2 = A @ Qb[:, k:]

    # obtain the reduced QR of A2
    Qa2, Ra2 = householder_qr(A2)

    # solve the least squares model for y_2
    y2 = scipy.linalg.solve_triangular(Ra2, Qa2.T @ (b - A1 @ y1))

    # substitute y1 and y2 back to get x

    # different outputs depending on whether y1 is a column vector or not
    if q == 1:
        x = Qb @ np.vstack((y1[:, np.newaxis], y2[:, np.newaxis]))
    else:
        x = Qb @ np.vstack((y1, y2))

    return x

plt.scatter(data[:,0] ,data[:,1], c="r", marker='x', label = "Observed Data") 

# generate evenly distributed values from 0 to 2
x_vals = np.arange(0,2.02,0.01)

p = 10
x = piecewise_poly(p,1)

y_vals = x_vals.copy()[:, np.newaxis]
y_vals[:101] = Vand(x_vals[:101], p+1) @ x[:p+1]
y_vals[101:] = Vand(x_vals[101:], p+1) @ x[p+1:]
plt.plot(x_vals[:101], y_vals[:101])
plt.plot(x_vals[100:], y_vals[100:])

# plt.show() ###################################################


RSS_mat = np.zeros(20)

for i in range(1,21):
    A = generate_mats(i)[0]
    x = piecewise_poly(i)
    RSS = np.sum ((A @ x - data[:,1][:, np.newaxis]) ** 2)
    RSS_mat[i-1] = RSS

plt.semilogy(range(1,21), RSS_mat)
# plt.show() ###################################################


"""
---------
PART D 
---------
"""
# generates a random parametric ellipse
def ellipse_with_noise(n, a, b):
    # generate random thetas and order them
    theta_i = np.sort(np.random.uniform(0,2*np.pi,(n)))

    # generate an ellipse with random noise using the theta values
    x_i = 5 + a * np.cos(theta_i) + np.random.multivariate_normal(np.zeros(n),np.eye(n)/5)
    y_i = 3 + b * np.sin(theta_i) + np.random.multivariate_normal(np.zeros(n),np.eye(n)/5)
        
    return theta_i, x_i, y_i

theta_i, x_i, y_i = ellipse_with_noise(100,20,10)

plt.scatter(x_i,y_i, c="r", marker='x', label = "Ellipse") 
# plt.show() ###################################################

# Fix M and p
M = 10
p = 5

# construct the required matrix A given theta values
def construct_A2(theta_i):
    A_mats = list()
    for m in range(M):
        # Obtain the index of theta values that are between 2*m*pi/M and 2*(m+1)*pi/M
        index = np.where(np.logical_and(theta_i>=2*np.pi*m/M, theta_i<2*np.pi*(m+1)/M))

        # Generate Vandermonde matrix for C_m
        A_mats.append(Vand(theta_i[index], p+1))
        

    # generate block diagonal matrix A
    A = A_mats[0]
    for i in range(1,M):
        A = scipy.linalg.block_diag(A, A_mats[i])

    return A

# construct the required matrix B given theta values
def construct_B2():
    # generate matrix B
    B = np.zeros((2*M,M*(p+1)))
    for i in range(M-1):
        # continuity
        B[i,i*(p+1):(i+1)*(p+1)] = np.array([1 * (2*np.pi*(i+1) / M)**i for i in range(p+1)])
        B[i,(i+1)*(p+1):(i+2)*(p+1)] = - B[i,i*(p+1):(i+1)*(p+1)]
                                                
        # continuity of derivatives
        B[i+M,i*(p+1):(i+1)*(p+1)] = np.multiply(np.array([1 * (2*np.pi*(i+1) / M)**(i-1) for i in range(p+1)]), np.arange(p+1))
        B[i+M,(i+1)*(p+1):(i+2)*(p+1)] = - B[i+M,i*(p+1):(i+1)*(p+1)]

    # edge case for continuity
    B[M-1,(M-1)*(p+1):M*(p+1)] = np.array([1 * (2*np.pi)**i for i in range(p+1)])
    B[M-1, 0] = -1

    # edge case for derivatives
    B[2*M-1,(M-1)*(p+1):M*(p+1)] = np.multiply(np.array([1 * (2*np.pi)**(i-1) for i in range(p+1)]), np.arange(p+1))
    B[2*M-1, 1] = -1

    return B

def generate_mats_2(p):
    A = construct_A2(theta_i)
    B = construct_B2()

    # generate matrix b
    b = np.concatenate((x_i[:, np.newaxis],y_i[:, np.newaxis]), axis=1)

    # generate matrix d
    d = np.zeros((2*M, 2))
    
    return A, B, b, d


