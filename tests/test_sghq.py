"""tests/test_sghq.py
Tests for sparse Gauss-Hermite Quadrature rule - Python implementation
Copyright (C) 2020 Eirik Ekjord Vesterkjaer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author:
    Eirik Ekjord Vesterkjaer
    eve.eirik@gmail.com
    github.com/eirikeve

Sources:
    [2] Jia, Bin: Sparse Gauss-Hermite Quadrature rule
        https://github.com/binjiaqm/sparse-Gauss-Hermite-quadrature-rule
        GitHub repository, Matlab code (commit 4afe0bc)

See README.md for more information, and LICENSE for the license.
"""

from pathlib import Path
from typing import Tuple
import pytest

import numpy as np

from sghq import sghq, sparsify_numerical_rule

def find_test_data(n, L):
    """Find the csv file for this case under data/
    The csv files were generated using generate_data.m, along with the code [2]
    """
    here = Path(__file__).parent.resolve()
    data_folder = Path(here, "data")

    path_X = Path(data_folder, f"SGHQ-X-L-{L}-n-{n}.csv").resolve()
    path_W = Path(data_folder, f"SGHQ-W-L-{L}-n-{n}.csv").resolve()

    if path_W.is_file() and path_X.is_file():
        X = np.genfromtxt(path_X, delimiter=',')
        W = np.genfromtxt(path_W, delimiter=',')

        X = X.reshape(-1, n)
        W = W.reshape(-1)

        return X, W
    raise FileNotFoundError()

def preprocess_data_so_tests_are_sane(X_m, W_m, X_py, W_py):
    """
    The tests of equality require that the arrays (X_m, X_py) and (W_m, W_py) are ordered identically,
    i.e. that each point / wt has the same index in both arrays.

    First of, np.unique can sort the arrays X_m, W_m to be the same as X_py, W_py

    However:
    The Matlab data tends to represent zeros as +/- 2.22E-16, which messes up the ordering from np.unique
    This makes it seem like the elements aren't the same when they are compared, as they are differently ordered.
    So also round the arrays to 12 significant digits, and then sort W_m / X_m using np.unique
    """
    X_m = np.around(X_m, decimals=12)
    W_m = np.around(W_m, decimals=12)
    X_py = np.around(X_py, decimals=12)
    W_py = np.around(W_py, decimals=12)

    # This doesn't change the points or weights as they are already sparse - it only sorts them (both)
    X_m, W_m = sparsify_numerical_rule(X_m, W_m)
    return X_m, W_m, X_py, W_py


@pytest.mark.parametrize("n", range(1, 7))
@pytest.mark.parametrize("L", range(1, 6))
def test_against_MATLAB_implementation(n, L):
    X_m, W_m = find_test_data(n, L)
    X_m, W_m = sparsify_numerical_rule(X_m, W_m)

    # This is the strategy I used when creating the test data,
    # and also the strategy that is default in the Matlab scripts of [2]
    X_py, W_py = sghq(n, L, strategy="third")

    # See docstring for more info on this:
    X_m, W_m, X_py, W_py = preprocess_data_so_tests_are_sane(X_m, W_m, X_py, W_py)

    assert X_m.shape == X_py.shape
    assert W_m.shape == W_py.shape

    err_X = X_m - X_py
    err_W = W_m - W_py

    assert np.sum(np.abs(err_W)) < 1e-9
    assert np.sum(np.abs(err_X)) < 1e-9
