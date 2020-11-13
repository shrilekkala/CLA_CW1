import numpy as np
from cla_utils import *
from matplotlib import pyplot as plt 

M = 100
x = np.arange(-M + 0.5, M + 0.5) / M
p = 5

# generate the required Vandermonde matrix
V = np.vander(x, p)
V = np.flip(V, axis = 1)

Q, R = householder_qr(V)

for i in range(5):
    plt.title("Data against Column " + str(i) + " of Q")
    plt.plot(x, Q[:,i])
plt.show()

for i in range(5):
    plt.title("Column " + str(i) + " of Q against the Data")
    plt.plot(Q[:,i], x)
plt.show()

