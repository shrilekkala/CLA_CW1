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

def generate_mats(p):
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
def piecewise_poly(p):
    A, B, b, d = generate_mats(p)

    # obtain the full QR of B^T
    Qb, Rb = householder_full_qr(B.T)
    Rb = Rb[:2,:2]

    # solve for y_1 by back substitution
    y1 = scipy.linalg.solve_triangular(Rb.T, d)

    A1 = A @ Qb[:, :2]
    A2 = A @ Qb[:, 2:]

    # obtain the reduced QR of A2
    Qa2, Ra2 = householder_qr(A2)

    # solve the least squares model for y_2
    y2 = scipy.linalg.solve_triangular(Ra2, Qa2.T @ (b - A1 @ y1))

    # substitute y1 and y2 back to get x
    x = Qb @ np.vstack((y1[:, np.newaxis], y2[:, np.newaxis]))

    return x

plt.scatter(data[:,0] ,data[:,1], c="r", marker='x', label = "Observed Data") 

# generate evenly distributed values from 0 to 2
x_vals = np.arange(0,2.02,0.01)

p = 10
x = piecewise_poly(p)

y_vals = x_vals.copy()[:, np.newaxis]
y_vals[:101] = Vand(x_vals[:101], p+1) @ x[:p+1]
y_vals[101:] = Vand(x_vals[101:], p+1) @ x[p+1:]
plt.plot(x_vals[:101], y_vals[:101])
plt.plot(x_vals[100:], y_vals[100:])

plt.show()


RSS_mat = np.zeros(20)

for i in range(1,21):
    A = generate_mats(i)[0]
    x = piecewise_poly(i)
    RSS = np.sum ((A @ x - data[:,1][:, np.newaxis]) ** 2)
    RSS_mat[i-1] = RSS

plt.semilogy(range(1,21), RSS_mat)
plt.show()
