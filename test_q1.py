''' test script for q1 '''
import pytest
import cla_utils
import q1

from numpy import random
import numpy as np

### test the accuracy of the QR factorisation ###
def test_unitary_Q():
    Q = q1.Q
    R = q1.R
    A = q1.A
    err = A - Q@R
    assert(np.linalg.norm(err) < 1.0e-6)