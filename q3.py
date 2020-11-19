import numpy as np
from cla_utils import *
from matplotlib import pyplot as plt 

def Vand(p):
    """
    Given a polynomial degree p
    construct the Vandermonde matrix V for use in the linear system Vx = b
    where x is the global variable of data points
 
    :return A: the matrix V in the linear system
    """
    V = np.vander(x, p+1)
    V = np.flip(V, axis = 1)

    return V

def plot_col_q(Q):
    """
    Given a matrix Q
    For each of the last 5 columns of Q,
    generate the plot of the column against the x (the global variable of data points)
 
    """
    for i in range(5):
        plt.plot(x, Q[:,-5+i], label = "Col." + str(i+1))

    plt.xlabel("Element of x")
    plt.ylabel("Element of Column of Q")


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""" Question 3a) """

M = 500
# create the x vector of equispaced points as required for a given M
x = np.arange(-M + 0.5, M + 0.5) / M


p = 5
# compute the reduced QR factorisation of V using householder
Q, R = householder_qr(Vand(p))
plt.title("Plot of Columns of Q against x (Householder)")

# plot column 0
plt.plot(x, Q[:,0], label = "Col." + str(0))

# plot the remaining 5 columns
plot_col_q(Q)
plt.legend(fontsize=8, loc='upper left')
plt.show()


""" Question 3b) """

# Invesigate QR factorisation for a large p
p = 100

# Householder
Qh, Rh = householder_qr(Vand(p))

plt.title("Plot of last 5 Columns of Q against x. (Householder), p = " + str(p) + ", M = " + str(M))
plot_col_q(Qh)
plt.show()

# Classical Gram-Schmidt
Qc, Rc = GS_classical(Vand(p))

plt.title("Plot of last 5 Columns of Q against x. (Classical GS), p = " + str(p) + ", M = " + str(M))
plot_col_q(Qc)
plt.show()

# Modified Gram-Schmidt
Qm, Rm = GS_modified(Vand(p))

plt.title("Plot of last 5 Columns of Q against x. (Modified GS), p = " + str(p) + ", M = " + str(M))
plot_col_q(Qm)
plt.show()
