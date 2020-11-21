''' test script for q1 '''
import pytest
import cla_utils
import q1
import numpy as np

# import variables from q1.py
Q = q1.Q
R = q1.R
A = q1.A

''' 
Test the accuracy of the QR factorisation
'''
def test_QR_accuracy():
    err = A - Q@R
    assert(np.linalg.norm(err) < 1.0e-6)


''' 
Test for the orthogonality of the Q matrix
'''
def test_QR_orthogonality():
    err = np.eye(Q.shape[1]) - Q.T @ Q
    assert(np.linalg.norm(err) < 1.0e-6)


''' 
Test that the R matrix is upper triangular to a threshold of 1.0x10^-6
'''
def test_R_triangular():
    assert(np.allclose(np.triu(R), R, 1.0e-6))


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)