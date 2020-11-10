from numpy import *
from cla_utils import *

A = load('values.npy')

# compute the QR factorisation of A using the modifies Gram-Schmidt algorithm
Q, R = GS_modified(A)