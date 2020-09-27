"""sghq/sghq.py
Sparse Gauss-Hermite Quadrature rule - Python implementation
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
    [1] Jia, Bin, Ming Xin, and Yang Cheng.
        "Sparse Gauss-Hermite quadrature filter with application to spacecraft attitude estimation."
        Journal of Guidance, Control, and Dynamics 34.2 (2011): 367-379.

    [2] Jia, Bin: Sparse Gauss-Hermite Quadrature rule
        https://github.com/binjiaqm/sparse-Gauss-Hermite-quadrature-rule
        GitHub repository, Matlab code (commit 4afe0bc)

See README.md for more information, and LICENSE for the license.
"""

from functools import lru_cache
from itertools import chain
from typing import Tuple, Sequence, List, Union, Callable

import numpy as np
from scipy.special import binom, roots_hermitenorm
from sympy.utilities.iterables import multiset_permutations, multiset


def sghq(n: int, L: int, strategy: Union[int, str, Callable[[int],int]]="third") -> Tuple[np.ndarray, np.ndarray]:
    """Sparse Gauss-Hermite Quadrature implementation

    Args:
        n (int): Dimensionality of the grid points.
        L (int): Accuracy level of the SGHQ grid.
        strategy (Union[int, str, Callable[int,int]]): Point selection strategy for the number m_L of univariate GHQ points for a given accuracy level L. Defaults to 3. Choices include "first", "second", "third", and their aliases 1, 2, 3. These correspond to the strategies for choosing m_L given in [1], Table (1): L, 2L-1, and 2**L-1 respectively. A Callable[int, int] can also be supplied.

    Returns:
        (np.array[None, n], np.array[None,]):  R^n points and their associated weight for the SGHQ(n, L)
    """
    strategy_func = resolve_point_selection_strategy(strategy)

    chi, W = [], []
    for q in range(L-n, L):
        Nn_q = N(n, q)
        for i, seq in enumerate(Nn_q):
            pts, unscaled_wts = form_grid(seq, strategy_func)
            wts = scale_weights(unscaled_wts, L, q, n)
            chi.append(pts)
            W.append(wts)
    # Same as the return values from [1], Algorithm (1), but we haven't accounted for duplicate points yet
    chi, W = np.concatenate(chi, axis=0), np.concatenate(W, axis=0)
    # So, post-process / handle the IF-statement in [1], Algorithm (1)
    return sparsify_numerical_rule(chi, W)



def resolve_point_selection_strategy(strategy: Union[int, str, Callable[[int],int]]):
    """Point selection strategy to choose the number of univariate GHQ points for a given accuracy level.
    This corresponds to m_L in [1], Table (1).
    """
    if callable(strategy):
        return strategy

    m_L_strategies = {
        1: lambda L: L,
        2: lambda L: 2*L - 1,
        3: lambda L: 2**L - 1,
        "first":  lambda L: L,
        "second": lambda L: 2*L - 1,
        "third":  lambda L: 2**L - 1,
    }
    if strategy not in m_L_strategies:
        raise ValueError(f"Expected strategy to be a Callable[int, int] or one of {m_L_strategies.keys()}, but got {strategy}")
    return m_L_strategies[strategy]

def form_grid(accuracy_sequence: Sequence[int], strategy_func: Callable[[int],int]) -> np.array:
    """Create a meshgrid of Gauss-Hermite quadrature points
    With accuracy along each dimension as given by the entries in accuracy_sequence.
    This implements the inner statement of [1], eq. (29), i.e. what's referred to as a tensor product sequence there.

    Args:
        accuracy_sequence (Sequence[int]): A sequence of n levels of accuracy, each indicating the accuracy of a Gauss-Hermite rule alonng a dimension.
        strategy_func (Callable[[int],int]): Function that implements point selection strategy for the number m_L of univariate GHQ points for a given accuracy level L.

    Returns:
        (np.array[None, n]): A set of n-dimensional points.
            The coarseness of the point coordinates across the i-th dimension is given by the i-th entry in the accuracy_sequence.
    """
    ms = [strategy_func(L) for L in accuracy_sequence]
    # GHQ (x, w) for all entries in the accuracy sequence.
    points_weights = [ghq( m ) for m in ms]
    points, weights = zip(*points_weights)
    points = [p.reshape(-1) for p in points]
    weights = [w.reshape(1, -1) for w in weights]



    dims = tuple([p.shape[-1] for p in points])
    nd = len(dims)
    # [1] refers to this procedure as a tensor product sequence, which I don't understand why.
    # I'd refer to it as making a high-dimensional meshgrid.
    # This part took a lot of time to understand
    nd_pt_grid = np.zeros( dims + (nd,) )
    nd_wt_grid = np.zeros( dims + (nd,) )
    for d_idx, (pt, wt) in enumerate(zip(points, weights)):
        # Repeat the coordinates of this point along the dimensions associated with other points.
        # This handles a single X_i in the inner U of [1] eq. (29)

        # View the point in the nd space, along its own dimension
        shape =[1 for d in dims] + [1]
        shape[d_idx] = len(pt)
        pt = pt.reshape(shape)
        wt = wt.reshape(shape)

        # Tile the point along the oher other dimensions
        repeat_counts = dims[:d_idx] + (1,) + dims[d_idx+1:] + (1,)
        repeated_1d_pt = np.tile(pt, repeat_counts)
        # The i-th dimension of the sparse grid quadrature contains the coordinates of the associated Gauss-Hermite quadrature points (given by the accuracy sequence entry)
        repeated_1d_wt = np.tile(wt, repeat_counts)

        nd_pt_grid[..., d_idx] = repeated_1d_pt[...,0]
        nd_wt_grid[..., d_idx] = repeated_1d_wt[...,0]

    nd_wt_grid = nd_wt_grid.reshape(-1, nd)

    unscaled_seq_wts= np.prod(nd_wt_grid, axis=-1)

    seq_grid = nd_pt_grid.reshape(-1, nd)

    return seq_grid, unscaled_seq_wts

def scale_weights(weights: np.array, L: int, q: int, n: int):
    """ Implements everything before the Prod in [1] eq. (30)
    """
    C = binom(n-1, L-1-q)
    return (-1)**(L - 1 - q) * C * weights

def sparsify_numerical_rule(points: np.array, weights: np.array):
    """Merges duplicate points, and sums their associated weights
    """
    sparse_grid, new_idcs = np.unique(points, return_inverse=True, axis=0)
    old_idcs = np.arange(new_idcs.shape[0])
    # This is a bit clumsy, but at least it's vecorized
    weight_map = np.zeros((sparse_grid.shape[0], points.shape[0]), dtype=weights.dtype)
    # Represent the summation of weights as a matrix multiplication
    weight_map[new_idcs, old_idcs] = 1
    sparse_weights = weight_map @ weights
    return sparse_grid, sparse_weights

@lru_cache(maxsize=32)
def ghq(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite quadrature with m evaluation points, for weighting function (1/2pi)^(1/2) e^(x^2 / 2)
    Can accurately propagate Gaussian uncertainties through polynomial transfer functions of deg <= 2,m-1

    Args:
        m (int): Number of evaluation points in GHQ grid

    Returns:
        Tuple[np.ndarray, np.ndarray]: (points, weights) of the quadrature rule
    """

    # Integal f(x) * e^(x^2 / 2) dx ~= Sum_i f(x) w_i
    x, w = roots_hermitenorm(m)
    x = x.reshape(-1)
    # Univariate standard Gaussian pdf is (1/2pi)^(1/2) e^(x^2 / 2), so scale weights accordingly
    w = w.reshape(-1) / (2 * np.pi)**(1/2)
    return x, w


def accuracy_level_combinations(n: int, q: int):
    """ Returns the set of sequences [i_1, ...] whose entries i_j sum to q and that are all of length <= n
    Used to implement [1] eq. (27)
    See just  below [1], eq. (30) where N_1^n is described.
    We can express it as all unique orderings of [1, 0, ..., 0] \in R^n, summed with [1, 1, ..., 1] \in R^n
    This function finds all the unique combinations which can be used to describe the unique orderings of the first array,

    Args:
        q (int): [description]
        n (int): [description]

    Returns:
        [type]: [description]

    Examples:
        ```python
        # For q = 0 and n >= 1 the return value will be
        [(0,)]
        # For q = 1 and n >= 1 the return value will be
        [(1, )]
        # for q = 2 and n >= 2 the return value will be
        [(2, ), (1, 1)]
        # for q = 3 and n >= 3 the return value will be
        [(3,), (2, 1), (1, 1, 1)]
        # for q = 3 and n == 2 the return value will be
        [(3,), (2, 1)]
        ```
    """
    if not type(q) is int or not type(n) is int:
        raise ValueError(f"Expected type {int} for d and n but but got type(q)={type(q)} and type(n)={type(n)}")
    if n < 1:
        raise ValueError(f"Expexted n >= 1, but got n = {n}")
    if q < 0:
        return []
    if q == 0:
        return [(0,)]
    return  _accuracy_level_combinations_impl(n, q)


@lru_cache(maxsize=128)
def _accuracy_level_combinations_impl(n: int, q: int):
    idcs = set()
    for k in range(q, 0, -1):
        if k == q:
            idcs.add( (q,) )
            continue
        new = set(
            (k,) + entry
            for entry in _accuracy_level_combinations_impl(n, q-k)
            if len(entry) < n
        )
        # this is useful, try accuracy_level_permutations(3, 3) with/without the line to see why
        new = set( ( tuple(sorted(entry, reverse=True)) for entry in new ) )
        idcs = idcs | new
    return sorted(idcs, reverse=True)

def expand_to_arrays(tuples: Sequence[Tuple], n: int):
    fill = 1
    return [
        np.array(
            [num + fill for num in tup] + [fill for i in range( n - len(tup) )]
        )
        for tup in tuples
    ]

@lru_cache(maxsize=16)
def N(n: int, q: int):
    """Constructs a set of accuracy level sequences, as given by [1] eq. (27)

    Args:
        n (int): [description]
        q (int): [description]

    Returns:
        np.array: (None, n) array, where n is the input n and the first dimension is the size
    """
    # prepare for [1], eq. (27). Get e.g. [(2,), (1,1)]
    accuracy_tuples = accuracy_level_combinations(n, q)
    # [(2,), (1,1)] -> [ [3, 1, ...], [2, 2, 1, ...] ]
    accuracy_seq_bases = expand_to_arrays(accuracy_tuples, n)
    # [1] eq. (27). [3, 1, ...] -> {[3,1, ...], [1, 3, ...], ...}, for each entry in the accuracy_tuples list
    # Xi is the weird square character used in [1] eq. (27)
    Xis = [multiset_permutations(seq) for seq in accuracy_seq_bases]
    # it's more practical to represent this as a np array.
    # as the array is nested, np.fromiter didn't seem to work.
    Nn_q = np.array(list(chain.from_iterable(Xis)), dtype=int)
    return Nn_q


if __name__ == "__main__":
    try:
        import fire
        notice = """sghq  Copyright (C) 2020 Eirik Ekjord Vesterkjaer
This program comes with ABSOLUTELY NO WARRANTY;
for details see the GPL-3 License: https://www.gnu.org/licenses/gpl-3.0
This is free software, and you are welcome to redistribute it under certain conditions;
for details see the GPL-3 License: https://www.gnu.org/licenses/gpl-3.0
"""
        print(notice)
        fire.Fire()
    except ModuleNotFoundError:
        print("run\n\npip install fire\n\nor\n\nconda install -c conda-forge fire\n\nto use the CLI")
