"""sghq/quadrature.py
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
from typing import Callable, Tuple, Union

import numpy as np
from scipy.special import roots_hermitenorm

from sghq.smolyak import sparse_grid

def sghq(n: int, L: int, strategy: Union[int, str, Callable[[int],int]]="third") -> Tuple[np.ndarray, np.ndarray]:
    """Sparse Gauss-Hermite Quadrature implementation

    Args:
        n (int): Dimensionality of the grid points.
        L (int): Accuracy level of the SGHQ grid.
        strategy (Union[int, str, Callable[int,int]]): Point selection strategy for the number m_L of univariate GHQ points for a given accuracy level L. Defaults to 3. Choices include "first", "second", "third", and their aliases 1, 2, 3. These correspond to the strategies for choosing m_L given in [1], Table (1): L, 2L-1, and 2**L-1 respectively. A Callable[int, int] can also be supplied.

    Returns:
        (Tuple[np.array, np.array]): Evaluation points and weights of the sparse grid (X, W), shaped [None, n] and [None,] respectively.
    """
    strategy_func = resolve_point_selection_func(strategy)
    def ghq_with_strategy(L):
        return ghq( strategy_func(L) )

    return sparse_grid(n, L, ghq_with_strategy)


def resolve_point_selection_func(strategy: Union[int, str, Callable[[int],int]]):
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

if __name__ == "__main__":
    try:
        import fire
        notice = """sghq.quadrature  Copyright (C) 2020 Eirik Ekjord Vesterkjaer
This program comes with ABSOLUTELY NO WARRANTY;
for details see the GPL-3 License: https://www.gnu.org/licenses/gpl-3.0
This is free software, and you are welcome to redistribute it under certain conditions;
for details see the GPL-3 License: https://www.gnu.org/licenses/gpl-3.0
"""
        print(notice)
        fire.Fire()
    except ModuleNotFoundError:
        print("run\n\npip install fire\n\nor\n\nconda install -c conda-forge fire\n\nto use the CLI")
