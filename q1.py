import numpy as np
from cla_utils import *
import matplotlib.pyplot

A = load('values.npy')

# compute the QR factorisation of A using the modified Gram-Schmidt algorithm
Q,R =  GS_modified(A)

# investigate the orthogonality of Q
err = Q.T @ Q - np.eye(Q.shape[1])
print(np.linalg.norm(err))

# investigate the diagonal components of R
matplotlib.pyplot.semilogy(R.diagonal(0))
matplotlib.pyplot.show()

# check the condition number of A
print(np.linalg.cond(A))
