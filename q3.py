import numpy as np
from cla_utils import *
from matplotlib import pyplot as plt 

# function that generates the required Vandermonde matrix given p
def Vand(p):

    V = np.vander(x, p)
    V = np.flip(V, axis = 1)

    return V

# function that generates the plot of x against the last 5 columns of Q
def plot_col_q(Q):

    for i in range(5):
        plt.plot(x, Q[:,i], label = "Col." + str(i+1))

    plt.legend(fontsize=8, loc='upper left')
    plt.xlabel("Column of Q")
    plt.ylabel("x")


# create the required x vector
M = 100
x = np.arange(-M + 0.5, M + 0.5) / M


p = 5
Q, R = householder_qr(Vand(p))

plt.title("Plot of Data x against Columns of Q (Householder)")
plot_col_q(Q)
plt.show()


# Invesigate QR factorisation for a large p
p = 100

# Householder
Qh, Rh = householder_qr(Vand(p))

plt.title("Plot of Data x against the last 5 Columns of Q (Householder)")
plot_col_q(Qh)
plt.show()

# Classical Gram-Schmidt
Qc, Rc = GS_classical(Vand(p))

plt.title("Plot of Data x against the last 5 Columns of Q (Classical GS)")
plot_col_q(Qc)
plt.show()

# Modified Gram-Schmidt
Qm, Rm = GS_modified(Vand(p))

plt.title("Plot of Data x against the last 5 Columns of Q (Modified GS)")
plot_col_q(Qm)
plt.show()
