sparse Gauss-Hermite Quadrature python implementation
# python-sghq

Python implementation of Smolyak's sparse grid method \[[2](#reference2)\] and the sparse Gauss-Hermite quadrature (SGHQ) algorithm \[[1](#reference1)\]. The SGHQ algorithm is used to obtain a numerical rule that approximates integrals over functions with Gaussian kernels.

## Installation
Code available at [https://github.com/eirikeve/python-sghq](https://github.com/eirikeve/python-sghq).

From GitHub:
```bash
git clone https://github.com/eirikeve/python-sghq
cd python-sghq
pip install -r requirements.txt
pip install -e .
```

## Usage
The SGHQ algorithm can be used by calling the function `X, W = sghq(n, L, [strategy])`, which is available in `sghq.quadrature`.
This yields evaluation points and weights for integration weighted by a _N(0, I)_ multivariate standard Gaussian.  They can be used similarly to the points and weights of the Unscented Transform, by first transforming them to match the multivatiate Gaussian you want to integrate over - see \[[1](#reference1)\].  

**Arguments:**  
- `n`: dimensionality of the grid points. E.g., for a 3-d state space, use `n=3`.
- `L`: accuracy level of the integration. The result will be exact for polynomials of order `<= 2L-1`.
- `strategy`: the selection strategy for univariate GHQ points for a given accuracy `L`. This has an impact on the total number of points in the sparse grid. You can choose between the following,
  - `1` (alias `"first"`): `m = L`
  - `2` (alias `"second"`): `m = 2*L - 1` (this is the default)
  - `3` (alias `"third"`):  `m = 2^L - 1` 
  -  or supply your own using a `lambda`.

The paper \[[1](#reference1)\], section (VI.C) indicates that the algorithm might be less sensitive to the `strategy` for larger values of `L`.


Some other functions are available in `sghq.smolyak` - e.g. `sparse_grid` which can be used to create sparse numerical rules based on other quadratures. `sqhq.quadrature.sghq` just wraps `sghq.smolyak.sparse_grid` with the Gauss-Hermite quadrature.


### Python example:

```python
> python
>>> import sghq.quadrature as quad
>>> n = 2
>>> L = 2
>>> strategy = 3
>>> X, W = quad.sghq(n, L, strategy=strategy)
>>> X
array([[-1.73205081,  0.        ],
       [ 0.        , -1.73205081],
       [ 0.        ,  0.        ],
       [ 0.        ,  1.73205081],
       [ 1.73205081,  0.        ]])
>>> W
array([0.16666667, 0.16666667, 0.33333333, 0.16666667, 0.16666667])
```

### CLI example:
If you have [`python-fire`](https://github.com/google/python-fire) installed you can also use a CLI to test things out:
```bash
# in python-sghq/ folder
> python sghq/quadrature.py sghq -n 2 -L 2 --strategy=3
(array([[-1.73205081,  0.        ],        [ 0.        , -1.73205081],        [ 0.        ,  0.        ],        [ 0.        ,  1.73205081],        [ 1.73205081,  0.        ]]), array([0.16666667, 0.16666667, 0.33333333, 0.16666667, 0.16666667]))
```

## Documentation 

See docstrings in the code.
These can be accessed using `help()`, e.g.:
```python
> python
>>> import sghq.quadrature as quad
>>> import sghq.smolyak as smol
>>> # Help for entire module, which also documents other available functions
>>> help(quad)
>>> help(smol)
>>> # Or just for one function:
>>> help(quad.sghq)
```

They can also be accessed through a CLI if you have `python-fire`:
```bash
# Help for entire module, which also documents other available functions
# Though this seems to show some includes as well
python sghq/quadrature.py -h
python sghq/smolyak.py -h
# Or just for one function:
python sghq/quadrature.py sghq -h
```

## Tests

There are some tests included. They test the implementation against data generated using Matlab codes by \[[2](#reference2)\], and run a few other small checks.
```bash
# in python-sghq/ folder
pytest
```

## About 

This implementation is based on the papers \[[1](#reference1)\] and \[3](#reference3)[\]
The Matlab implementation \[[2](#reference2)\] of \[[1](#reference1)\] by Bin Jia (one of the authors) was also used as a reference, but primarily for debugging and comparing results. The test data was generated using that code.

I implemented this because I couldn't locate an existing implementation of the SGHQ for Python, and failed to get a Matlab-to-Python transpiler working.

It appears to run at least as fast as the Matlab implementation \[[2](#reference2)\] (though I tested \[[2](#reference2)\] using Octave - so I'm not sure about how that extrapolates to Matlab). For the tests I've ran it yields identical results to those codes.

### Theory
The SGHQ is a numerical rule that can be used for integrating functions with a Gaussian kernel.

It's similar to the Unscented Transform, but can support higher levels of accuracy (though at a higher computational cost, though still under the assumption of Gaussian uncertainties).  

The SGHQ(_n_, _L_) alorithm creates a set of _n_-dimensional points and associated _1_-dimensional weights. These can approximate the integral over a function _f: n -> m_ weighted by a standard multivariate Gaussian , _X ~ N(x; 0, I)_. The _n_ random variables in _X_ are i.i.d. _xi ~(1/2pi)^(1/2) * e^(xi^2 / 2)_. The accuracy level _L_ determines to what order of polynomial _f_ the integration is accurate for. For a given _L_, integration over a polynomial _f_ of up to order _2L-1_ will be exact.  


Any multivariate Gaussian can be expressed in terms of a standard multivariate Gaussian  and an affine transformation (see e.g. \[[1](#reference1)\], eq. (23)). This means that the SGHQ rule can be applied for any (non-degenerate) multivariate Gaussian. This makes it suitable for nonlinear Gaussian filtering tasks - see e.g. \[[1](#reference1)\] or \[[2](#reference1)\] for more in-depth information.

## References

> **[1]** <a name="reference1"></a> Jia, Bin; Ming Xin; Yang Cheng. "Sparse Gauss-Hermite quadrature filter with application to spacecraft attitude estimation" Journal of Guidance, Control, and Dynamics Vol. 32, no. 2 (2011). \[[PDF](https://www.researchgate.net/publication/258837425_Sparse_Gauss-Hermite_Quadrature_Filter_with_Application_to_Spacecraft_Attitude_Estimation)\]

> **[2]**  <a name="reference2"></a> Jia, Bin (binjiaqm). "Sparse Gauss-Hermite Quadrature rule". GitHub repository with Matlab code (commit 4afe0bc). \[[Repo](https://github.com/binjiaqm/sparse-Gauss-Hermite-quadrature-rule)\]


> **[3]** <a name="reference3"></a> Heiss, Florian, and Viktor Winschel. "Likelihood approximation by numerical integration on sparse grids." Journal of Econometrics 144.1 (2008). \[[PDF](https://hal.archives-ouvertes.fr/hal-00501810/)\]
 
## License and Contact

[GNU General Public License Version 3](LICENSE).

[eve.eirik@gmail.com](mailto:eve.eirik@gmail.com)  
[https://github.com/eirikeve](https://github.com/eirikeve)  
Code available at: [https://github.com/eirikeve/python-sghq](https://github.com/eirikeve/python-sghq)  
