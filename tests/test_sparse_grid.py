import numpy as np
import pytest
from scipy.special import roots_legendre, roots_laguerre

from sghq.quadrature import ghq, resolve_point_selection_func
from sghq.smolyak import quadrature_tensor_product, sparse_grid, sparsify_numerical_rule

from .test_quadrature import find_test_data, preprocess_data_so_tests_are_sane


@pytest.mark.parametrize("n", range(1, 7))
@pytest.mark.parametrize("L", range(2, 6))
@pytest.mark.parametrize("quadrature", (ghq, roots_legendre, roots_laguerre))
def test_sparse_and_dense_grid_statistics(n, L, quadrature):
    """The statistics of the sparse and dense quadrature rules should be exactly the same."""

    dense_Xi = [L for i in range(n)]
    X_dense, W_dense = quadrature_tensor_product(dense_Xi, quadrature)
    X_sparse, W_sparse = sparse_grid(n, L, quadrature)

    # Statistics of the distributions should be equal
    # both with mean == [0, 0, ..., 0] and cov == diag([1, 1, ..., 1])
    mu_dense = X_dense.T @ W_dense
    mu_sparse = X_sparse.T @ W_sparse

    assert np.allclose(mu_dense, mu_sparse)

    dev_dense = X_dense - mu_dense.reshape(1, -1)
    dev_sparse = X_sparse - mu_sparse.reshape(1, -1)

    Sigma_dense = (dev_dense * W_dense.reshape(-1, 1)).T @ dev_dense
    Sigma_sparse = (dev_sparse * W_sparse.reshape(-1, 1)).T @ dev_sparse

    assert np.allclose(Sigma_dense, Sigma_sparse)


@pytest.mark.parametrize("n", (32, 64))
@pytest.mark.parametrize("L", (2, 3))
@pytest.mark.parametrize("quadrature", (ghq,))
def test_high_dim_sparse_grid_statistics(n, L, quadrature):
    # A dense grid for such high dimensional data can't be allocated
    # as it requires L^n * nd floating point values (assuming m_L == L)
    # So instead, just ensure the statistics are as expected.
    # Which we know should be N(0, I) for the SGHQ
    X_sparse, W_sparse = sparse_grid(n, L, quadrature)
    mu_sparse = X_sparse.T @ W_sparse
    zeros = np.zeros_like(mu_sparse)

    assert np.allclose(zeros, mu_sparse)

    dev_sparse = X_sparse - mu_sparse.reshape(1, -1)

    Sigma_sparse = (dev_sparse * W_sparse.reshape(-1, 1)).T @ dev_sparse
    Identity = np.diag(np.ones_like(mu_sparse))

    assert np.allclose(Identity, Sigma_sparse)


@pytest.mark.parametrize("n", range(1, 7))
@pytest.mark.parametrize("L", range(1, 6))
def test_sparse_grid_against_MATLAB_data(n, L):
    """This is sort of a duplicate of test_against_MATLAB_implementation in test_sghq.py
    Might be useful to explicitly test both cases in the event that one function changes though.
    """
    X_m, W_m = find_test_data(n, L)
    X_m, W_m = sparsify_numerical_rule(X_m, W_m)

    # This is the strategy I used when creating the test data,
    # and also the strategy that is default in the Matlab scripts of [2]
    strategy = resolve_point_selection_func(strategy="third")
    X_sm, W_sm = sparse_grid(n, L, lambda L: ghq(strategy(L)))
    # See docstring for more info on this:
    X_m, W_m, X_sm, W_sm = preprocess_data_so_tests_are_sane(X_m, W_m, X_sm, W_sm)

    assert X_m.shape == X_sm.shape
    assert W_m.shape == W_sm.shape

    err_X = X_m - X_sm
    err_W = W_m - W_sm

    assert np.sum(np.abs(err_W)) < 1e-9
    assert np.sum(np.abs(err_X)) < 1e-9
