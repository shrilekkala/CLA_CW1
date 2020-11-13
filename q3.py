import numpy as np
from cla_utils import *

M = 2
x = np.arange(-M + 0.5, M + 0.5) / M
p = 5

# generate the required Vandermonde matrix
V = np.vander(x, p)
V = np.flip(V, axis = 1)