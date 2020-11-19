import numpy as np
from cla_utils import *
from matplotlib import pyplot as plt 

A = np.load('values.npy')

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


""" Question 1a) """

# compute the QR factorisation of A using the modified Gram-Schmidt algorithm from 
Q,R =  GS_modified(A)

""" Question 1b) """

# investigate the orthogonality of Q
err = Q.T @ Q - np.eye(Q.shape[1])
print("The error in the orthogonality of Q is", np.linalg.norm(err))

# investigate the diagonal components of R
plt.semilogy(R.diagonal(0))
plt.title("Semilog plot of the diagonal components of R")
plt.xlabel("n")
plt.ylabel("n'th component of diagonal of R")
plt.show()

# plot the columns of R
plt.title("Plot of the columns of R")
for i in range(100):
    plt.plot(R[:,i])
plt.xlabel("n")
plt.ylabel("n'th element of a column of R")
plt.show()


