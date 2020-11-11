from numpy import *
from cla_utils import *

A = load('values.npy')

# compute the QR factorisation of A using the modified Gram-Schmidt algorithm
Q,R =  GS_modified(A)

# investigate the orthogonality of Q
err = Q.T @ Q - np.eye(Q.shape[1])
print(np.linalg.norm(err))