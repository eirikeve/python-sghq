sparse Gauss-Hermite Quadrature python implementation
# python-sghq

Python-implementation of the sparse Gauss-Hermite quadrature (SGHQ) algorithm. The SGHQ is used to obtain a numerical rule to integrate functions with Gaussian kernels.

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
The algorithm can be used by calling the function `sghq(n, L, [strategy])`, which is available in `sghq.sghq` (yes).

Some other functions are available in `sghq.sghq` - these probably won't be especially interesting (except if you want to know how things are being done). 


The `n` parameter is the dimensionality of the grid points. E.g., for a 3-d state space, use `n=3`.

The `L` parameter decides how accurate the integration is. The result will be exact for polynomials of order `<= 2L-1`.

The `strategy` parameter decides the selection strategy for univariate SGHQ points for a given accuracy L. The paper \[[1](#reference1)\] seems to use the third variant. This has an impact on the total number of points in the sparse grid.
You can choose between the following,
* `1` (alias `"first"`): `m = L`
* `2` (alias `"second"`): `m = 2*L - 1` 
* `3` (alias `"third"`):  `m = 2^L - 1` (this is the default)
* Or supply your own using a `lambda`.

### Usage in Python code:
```python
import sghq.sghq as s
n = 2
L = 2
strategy = 3
X, W = s.sghq(n, L, strategy=strategy)
# X is a (None, n) np.array, W is a (None,) np.array
```

If you have [`python-fire`](https://github.com/google/python-fire) installed you can also use a CLI to test things out:
```bash
# in python-sghq/ folder
python sghq/sghq.py sghq -n 2 -L 2 --strategy=3
```

## Documentation 

See docstrings in the code.
These can be accessed using `help()`, e.g.:
```python
python
>>> import sghq.sghq as s
>>> # Help for entire module, which also documents other available functions
>>> help(s)
>>> # Or just for one function:
>>> help(s.sghq)
```

They can also be accessed through a CLI if you have `python-fire`:
```bash
# Help for entire module, which also documents other available functions
# Though this seems to show some includes as well
python sghq/sghq.py -h
# Or just for one function:
python sghq/sghq.py sghq -h
```

## Tests

Run supplied tests that compares algorithm output to data generated using Matlab codes by \[[2](#reference2)\]:
```bash
# in python-sghq/ folder
pytest
```

## About 

This implementation is based on the paper \[[1](#reference1)\].
The Matlab implementation \[[2](#reference2)\] of \[[1](#reference1)\] by Bin Jia (one of the authors) was also used as a reference, but primarily for debugging and comparing results. The test data was generated using that code.

I implemented this because I couldn't locate an existing implementation of the SGHQ for Python, and failed to get a Matlab-to-Python transpiler working.

It appears to run at least as fast as the Matlab implementation \[[2](#reference2)\], eq. (12) (though I tested it using Octave - so I'm not sure about how that extrapolates to Matlab). For the tests I've ran it yields identical results to those codes.

Still: No guarantees on the code efficiency or algorithm correctness :)

### Theory
The SGHQ is a numerical rule that can be used for integrating functions with a Gaussian kernel.


It's similar to the Unscented Transform (see \[[3](#reference3)\], eq. (12)), but can support higher levels of accuracy (though at a higher computational cost, and still under the assumption of Gaussian uncertainties).

This can be used in nonlinear Gaussian filtering tasks to propagate Gaussian state beliefs through nonlinear state transition functions and measurement functions, see \[[1](#reference1)\] for more in-depth information.

The SGHQ(_n_, _L_) alorithm creates a set of _n_-dimensional points and associated _1_-dimensional weights. These can approximate the integral over a function _f: n -> r in R^1_ weighted by a standard Gaussian probability density function, _N(x; 0, 1) = ((1/2pi)^(1/2))*e^(x^2 / 2)_. The accuracy level _L_ determines to what order of polynomial _f_ the integration is accurate for. For a given _L_, integration over a polynomial _f_ of up to order _2L-1_ will be exact.

## References

> **[1]** <a name="reference1"></a> Jia, Bin; Ming Xin; Yang Cheng. "Sparse Gauss-Hermite quadrature filter with application to spacecraft attitude estimation" Journal of Guidance, Control, and Dynamics Vol. 32, no. 2 (2011). \[[PDF](https://www.researchgate.net/publication/258837425_Sparse_Gauss-Hermite_Quadrature_Filter_with_Application_to_Spacecraft_Attitude_Estimation)\]

> **[2]**  <a name="reference2"></a> Jia, Bin (binjiaqm). "Sparse Gauss-Hermite Quadrature rule". GitHub repository with Matlab code (commit 4afe0bc). \[[Repo](https://github.com/binjiaqm/sparse-Gauss-Hermite-quadrature-rule)\]

> **[3]**  <a name="reference3"></a> Julier, Simon J.; Uhlmann, Jeffrey K. "Unscented Filtering and Nonlinear Estimation" Proceedings of the IEEE, Vol. 92, no. 3 (2004). \[[PDF](https://www.cs.ubc.ca/~murphyk/Papers/Julier_Uhlmann_mar04.pdf)\]

 
## License and Contact

[GNU General Public License Version 3](LICENSE).

[eve.eirik@gmail.com](mailto:eve.eirik@gmail.com)  
[https://github.com/eirikeve](https://github.com/eirikeve)  
Code available at: [https://github.com/eirikeve/python-sghq](https://github.com/eirikeve/python-sghq)  
