''' test script for q2 '''
import pytest
import cla_utils
import q2

from numpy import random
import numpy as np

# import variables from q1.py
x = q2.x
f = q2.f

### test the accuracy of the Least Squares Polynomial using RSS (Residual sum of squares) to a specified tolerance###
@pytest.mark.parametrize('m, tol', [(10, 1e-06), (10,1), (7, 1e-06), (7,1)])
def test_LS_accuracy(m, tol):
    
    A = q2.linear_system(m, x)
    coef = q2.least_squares(A, f)

    RSS = np.sum ((A @ coef - f) ** 2)
    assert(RSS < tol)