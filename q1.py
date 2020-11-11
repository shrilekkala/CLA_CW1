from numpy import *
from cla_utils import *

A = load('values.npy')
x = 5
# compute the QR factorisation of A using the modified Gram-Schmidt algorithm
def get_QR():
    return GS_modified(A)

Q,R = get_QR()

 