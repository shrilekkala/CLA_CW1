''' test script for q1 '''
import pytest
import cla_utils
import q1

from numpy import random
import numpy as np

# import variables from q1.py
Q = q1.Q
R = q1.R
A = q1.A

### test the accuracy of the QR factorisation ###
def test_QR_accuracy():
    err = A - Q@R
    assert(np.linalg.norm(err) < 1.0e-6)

### test if the Q is orthogonal ###
def test_QR_orthogonality():
    err = np.eye(Q.shape[1]) - Q.T @ Q
    assert(np.linalg.norm(err) < 1.0e-6)

### test if the R is upper triangular ###
def test_R_triangular():
    assert(np.allclose(np.triu(R), R, 1.0e-6))