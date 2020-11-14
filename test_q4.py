''' test script for q4 '''
import pytest
import cla_utils
import q4

import numpy as np

# import variables from q1.py
data = q4.data

### test the accuracy of the Least Squares Polynomial using RSS (Residual sum of squares) to a specified tolerance###
@pytest.mark.parametrize('p, tol', [(3, 1e-06), (3,10), (5, 1e-06), (5,10), (10, 1e-06), (10,10)])
def test_LS_accuracy_4c(p, tol):
        
    A = q4.generate_mats(p)[0]
    x = q4.piecewise_poly(p)

    RSS = np.sum ((A @ x - data[:,1][:, np.newaxis]) ** 2)
    assert(RSS < tol)