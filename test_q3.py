''' test script for q3 '''
import pytest
import cla_utils
import q3

from numpy import random
import numpy as np

# import variables from q1.py
Q = q3.Q
R = q3.R
V = q3.Vand(5)

### test the accuracy of the householder QR factorisation ###
def test_QR_accuracy():
    err = V - Q@R
    assert(np.linalg.norm(err) < 1.0e-6)

### test if the householder Q is orthogonal ###
def test_QR_orthogonality():
    err = np.eye(Q.shape[1]) - Q.T @ Q
    assert(np.linalg.norm(err) < 1.0e-6)

### test if the householder R is upper triangular ###
def test_R_triangular():
    assert(np.allclose(np.triu(R), R, 1.0e-6))