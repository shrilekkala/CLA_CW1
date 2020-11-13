import numpy as np
from cla_utils import *
import scipy

data = np.loadtxt('pressure.dat')

# function that generates the required Vandermonde matrix given some data p
def Vand(x, p):

    V = np.vander(x, p)
    V = np.flip(V, axis = 1)

    return V

p = 10

# formulate matrices A, B and vectors b, d from the data
A = np.zeros((100,2*(p+1)))
A[:51,:p+1] = Vand(data[:51,0], p+1)
A[51:,p+1:] = Vand(data[51:,0], p+1)

B = np.ones((2,2*(p+1)))
B[0, p+1:] = -np.ones(p+1)
B[1, :p+1] = np.arange(p+1)
B[1, p+1:] = -np.arange(p+1)

b = data[:,1]

d = np.array([0, -5]).T


# obtain the reduced QR of B^T
Qb, Rb = householder_qr(B.T)

# solve for y_1 by back substitution