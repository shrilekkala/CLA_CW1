''' test script for q4 '''
import pytest
import cla_utils
import q4
import numpy as np

# import variables from q1.py
data = q4.data


''' 
Test the accuracy of the Least Squares Polynomial in 4c) to a specified tolerance'
using RSS (Residual sum of squares)
'''
@pytest.mark.parametrize('p, tol_lower, tol_upper', [(3,1,10), (5,1,10), (10,1,10)])
def test_LS_accuracy_4c(p, tol_lower, tol_upper):

    A = q4.generate_mats_1(p)[0]
    x = q4.piecewise_poly(p, 2, 1)

    RSS = np.sum ((A @ x - data[:,1][:, np.newaxis]) ** 2)
    assert(tol_lower < RSS < tol_upper)


''' 
Test the accuracy of the Least Squares Polynomial for various closed curves in 4d) to a specified tolerance'
using RSS (Residual sum of squares)
'''
@pytest.mark.parametrize('M, p, a, b', [(5, 6, 20, 50), (10, 3, 60, 10), (2, 9, 25, 25)])
def test_LS_accuracy_4d(M, p, a, b):
    theta_i, x_i, y_i = q4.ellipse_with_noise(250,a,b)

    A, B, b, d = q4.generate_mats_2(p, M)
    x = q4.piecewise_poly(p, M, 2)

    err = np.linalg.norm(A @ x - b)
    assert(err < 50)