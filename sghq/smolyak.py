"""sghq/smolyak.py
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

    [3] Heiss, Florian, and Viktor Winschel.
        "Likelihood approximation by numerical integration on sparse grids."
        Journal of Econometrics 144.1 (2008): 62-80.

See README.md for more information, and LICENSE for the license.
"""

from functools import lru_cache
from itertools import chain
from typing import Callable, List, Tuple, Sequence, Union


import numpy as np

from scipy.sparse import coo_matrix
from scipy.special import binom
from sympy.utilities.iterables import multiset_permutations

def sparse_grid(n: int, L: int,
                quadrature: Callable[[int], Tuple[np.array, np.array]]
                ) -> Tuple[np.array, np.array]:
    """Smolyak's rule for n-d sparse-grid numerical integration based on a 1-d quadrature rule
    For numerical integration
    This implementation is based on [1], [2] and [3]

    Args:
        n (int): Dimensionality of the grid points.
        L (int): Accuracy level of the sparse grid.
        quadrature (Callable[[int], Tuple[np.array, np.array]]): Univariate quadrature rule, which accepts an accuracy level L as parameter and returns (pts, wts). For instance a quadrature rule from scipy.special.

    Returns:
        (Tuple[np.array, np.array]): Evaluation points and weights of the sparse grid (X, W), shaped [None, n] and [None,] respectively.
    """
    chi, W = [], []
    for q in range(L-n, L):
        Nn_q = N(n, q)
        for Xi in Nn_q:
            pts, unscaled_wts = form_grid(Xi, quadrature)
            wts = scale_weights(unscaled_wts, L, q, n)
            chi.append(pts)
            W.append(wts)
    chi, W = np.concatenate(chi, axis=0), np.concatenate(W, axis=0)
    return sparsify_numerical_rule(chi, W)

def N(n: int, q: int) -> Sequence[Tuple]:
    """Constructs a set of accuracy level sequences, as given by [1] eq. (27)

    Args:
        n (int): [description]
        q (int): [description]

    Returns:
        Sequence[Tuple]: tuples of lengths n, each corresponding to an accuracy sequence for n dimensions of the integration
    """
    # prepare for [1], eq. (27). Get e.g. [(2,), (1,1)]
    accuracy_tuples = accuracy_level_combinations(n, q)
    # [(2,), (1,1)] -> [ [3, 1, ...], [2, 2, 1, ...] ]
    accuracy_seq_bases = expand_to_length(accuracy_tuples, n)
    # [1] eq. (27). [3, 1, ...] -> {[3,1, ...], [1, 3, ...], ...}, for each entry in the accuracy_tuples list
    # Xi is the weird square character used in [1] eq. (27)
    Xis = [multiset_permutations(seq) for seq in accuracy_seq_bases]
    Nn_q = chain.from_iterable(Xis)
    return Nn_q

def form_grid(accuracy_sequence: Sequence[int],
              quadrature: Callable[[int], Tuple[np.array, np.array]]
              ) -> Tuple[np.array, np.array]:
    """Create a dense grid of quadrature points
    With accuracy along each dimension as given by the entries in accuracy_sequence.
    This implements the inner statement of [1], eq. (29), i.e. what's referred to as a tensor product sequence there.

    [3], eq. (6) is important for understanding this part. They define the tensor product of uv quadrature rules as,

    (V_{i_1} ⊗ ...  ⊗ V_{i_n})[f] = Σ_{x1 in X_{i,1}} ... Σ_{xn in X_{i,n}} f(x1, ..., xn) Π_{d = 1}^D w_{i_d}

    So this means that what [1] and [3] call a tensor product sequence of points and weights isn't a normal tensor product:
    It's just all the combinations of all individual quadrature points along their respective dimension (which are given by the index / position of the quadrature rule in the "tensor product").
    Knowing this, we can see that the point grid (X_1 ⊗ ... ⊗ X_n) can be expressed pretty easily,
    by simply repeating each X_i along all of the other quadrature rules' dimensions (which gives a (L_1, L_2, ..., L_n) tensor of 1d points )
    and assigning this resulting point sequence to the corresponding index of the last dimension resulting point set.
    The point set is a (L_1, L_2, ..., L_n, n) tensor, which is afterwards reshaped to (-1, n).
    The same goes for the weights - just that the last dimension of the tensor is reduced to the product of its entries.

    Args:
        accuracy_sequence (Sequence[int]): A sequence of n levels of accuracy, each indicating the accuracy of a Gauss-Hermite rule alonng a dimension.
        strategy_func (Callable[[int],int]): Function that implements point selection strategy for the number m_L of univariate quadrature points for a given accuracy level L.

    Returns:
        (np.array[None, n], np.array[None,]): A set of n-dimensional points and their weights. The coarseness of the point coordinates across the i-th dimension is given by the i-th entry in the accuracy_sequence.
    """
    points_weights = [ quadrature(L) for L in accuracy_sequence ]
    points, weights = zip(*points_weights)
    points = [p.reshape(-1) for p in points]
    weights = [w.reshape(-1) for w in weights]

    dims = tuple([p.shape[-1] for p in points])
    nd = len(dims)
    # This follows the approach used in [1], [2], [3] for expressing the tensor products in Smolyak's rule
    # I'd refer to it as making a high-dimensional meshgrid, not a tensor product.
    # This part took a lot of time to understand
    nd_pt_grid = np.zeros( dims + (nd,) )
    nd_wt_grid = np.zeros( dims + (nd,) )
    for d_idx, (pt, wt) in enumerate(zip(points, weights)):
        # Repeat the coordinates of this point along the dimensions associated with other points.
        # This handles a single X_i in the inner U of [1] eq. (29)
        # View the point in the nd space, along its own dimension
        shape =[1 for d in dims]
        shape[d_idx] = len(pt)
        pt = pt.reshape(shape)
        wt = wt.reshape(shape)

        # Tile the point along the oher other dimensions
        repeat_counts = dims[:d_idx] + (1,) + dims[d_idx+1:]
        repeated_1d_pt = np.tile(pt, repeat_counts)
        repeated_1d_wt = np.tile(wt, repeat_counts)

        # The i-th dimension of the grid from this accuracy sequence consists of the coordinates of the associated univariate quadrature points, the number of which (L) is given by the corresponding accuracy sequence (Xi) entry
        nd_pt_grid[..., d_idx] = repeated_1d_pt
        nd_wt_grid[..., d_idx] = repeated_1d_wt

    nd_wt_grid = nd_wt_grid.reshape(-1, nd)
    unscaled_seq_wts= np.prod(nd_wt_grid, axis=-1)
    seq_grid = nd_pt_grid.reshape(-1, nd)
    return seq_grid, unscaled_seq_wts

def scale_weights(weights: np.array, L: int, q: int, n: int) -> np.array:
    """ Implements everything before the Prod in [1] eq. (30)
    This weight scaling lets us express the sparse grid as a summation over univariate quadrature rules,
    Instead of as a summation over their differences. See [3], eq. (10)
    """
    C = binom(n-1, L-1-q)
    return (-1)**(L - 1 - q) * C * weights

def sparsify_numerical_rule(points: np.array, weights: np.array) -> Tuple[np.array, np.array]:
    """Merges duplicate points, and sums their associated weights

    Args:
        points (np.array[None, n]): array of n-dimensional quadrature rule points - possibly with duplicates.
        weights (np.array[None]): array of quadrature rule weights associated with the points.

    Returns:
        (np.array[None, n], np.array[None]): Sparse grid, without duplicate points. The weights have been updated to account for points that are merged.
    """
    # This is the bottleneck of the algorithm, didn't find any good alternatives
    sparse_points, new_idcs = np.unique(points, return_inverse=True, axis=0)
    old_idcs = np.arange(new_idcs.shape[0])
    ones = np.ones_like(new_idcs)
    # Using a dense matrix here leads to a memory usage explosion
    # as the number of (dense) points grows very rapidly.
    # for small numbers of points this might be a bit slower, but in those cases
    # the algorithm is pretty fast anyways.
    weight_map = coo_matrix((ones, (new_idcs, old_idcs)),
                            shape=(sparse_points.shape[0], points.shape[0]))
    sparse_weights = weight_map.dot(weights)

    return sparse_points, sparse_weights

def accuracy_level_combinations(n: int, q: int):
    """ Returns the set of sequences [i_1, ...] whose entries i_j sum to q and that are all of length <= n
    Used to implement [1] eq. (27)
    See just  below [1], eq. (30) where N_1^n is described.
    We can express it as all unique orderings of [1, 0, ..., 0] \in R^n, summed with [1, 1, ..., 1] \in R^n
    This function finds all the unique combinations which can be used to describe the unique orderings of the first array,

    Args:
        q (int): Integer > 0, ceiling for the accuracy in the sequences
        n (int): Integer > 1, max number of points in each sequence

    Returns:
        [type]: Set of unique k-dimensional sequences (k < n) whose entries sum to q

    Examples:
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


@lru_cache(maxsize=256)
def _accuracy_level_combinations_impl(n: int, q: int):
    """See accuracy_level_combinations"""
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
        # this is useful, try accuracy_level_combinations(3, 3) with/without the line to see why
        new = set( ( tuple(sorted(entry, reverse=True)) for entry in new ) )
        idcs = idcs | new
    return sorted(idcs, reverse=True)

def expand_to_length(tuples: Sequence[Tuple], n: int, fill=1) -> List[Tuple]:
    """Right-pads all entries in the sequence with fill such that they are of length n

    Args:
        tuples (Sequence[Tuple])
        n (int): length to expand tuples to

    Returns:
        List[Tuple]: Input tuples, but right padded with fill

    """
    return [
        tuple(
            [num + fill for num in tup] + [fill for i in range( n - len(tup) )]
        )
        for tup in tuples
    ]

if __name__ == "__main__":
    try:
        import fire
        notice = """sghq.smolyak  Copyright (C) 2020 Eirik Ekjord Vesterkjaer
This program comes with ABSOLUTELY NO WARRANTY;
for details see the GPL-3 License: https://www.gnu.org/licenses/gpl-3.0
This is free software, and you are welcome to redistribute it under certain conditions;
for details see the GPL-3 License: https://www.gnu.org/licenses/gpl-3.0
"""
        print(notice)
        fire.Fire()
    except ModuleNotFoundError:
        print("run\n\npip install fire\n\nor\n\nconda install -c conda-forge fire\n\nto use the CLI")
