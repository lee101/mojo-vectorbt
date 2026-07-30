# mojo-vectorbt

`mojo-vectorbt` is a standalone Mojo port of the compute-heavy NumPy/Numba
kernels behind [vectorbt](https://vectorbt.dev/). It keeps vectorbt's function
names, argument order, defaults, 1-D suffix convention, and axis-0 time-series
layout for the covered subset.

The package is intentionally named `mojo_vectorbt`, so it can be installed
beside the real `vectorbt` and parity-tested against it. Switching a covered
module requires only changing the import:

```python
import numpy as np
from mojo_vectorbt.returns import nb

returns = np.array([0.01, -0.01, 0.03])

print(nb.cum_returns_1d_nb(returns, 100.0))
# [101.      99.99   102.9897]
print(nb.max_drawdown_1d_nb(returns))
# -0.01
```

## Covered subset

There are 55 compatible Python functions.

| vectorbt module | implemented functions |
| --- | --- |
| `generic.nb` | `fshift_*`, `bshift_*`, `diff_*`, `pct_change_*`, `rolling_mean_*`, `rolling_std_*`, `rolling_min_*`, `rolling_max_*`, each in `_1d_nb` and matrix `_nb` form |
| `returns.nb` | `get_return_nb`; 1-D and matrix forms of `returns`, `cum_returns`, `cum_returns_final`, `drawdown`, `max_drawdown`, `annualized_return`, `annualized_volatility`, `calmar_ratio`, `omega_ratio`, `sharpe_ratio`, `downside_risk`, `sortino_ratio`, `information_ratio`, `beta`, `alpha`, `tail_ratio`, `value_at_risk`, and `cond_value_at_risk` |
| `signals.nb` | `clean_enex_1d_nb`, `clean_enex_nb` |

The parity tests exercise every function listed above directly against
vectorbt 1.1.0 using identical arrays. Coverage includes NaNs, multi-column
inputs, non-contiguous and integer inputs, window and `minp` combinations,
`ddof`, equity crossing zero, simultaneous signals, quantile interpolation,
SIMD tails, parallel quantiles, and empty-reduction behavior.

This is not a port of vectorbt's pandas accessors, plotting, data downloaders,
record classes, indicator factory, portfolio object, order simulator,
callback-based kernels, or every rolling performance metric. Inputs to the
covered numeric API are converted to C-contiguous `float64` arrays. Complex
values, floating-point types wider than `float64`, and integers outside
`float64`'s exact range are rejected rather than silently narrowed. The generic
shift functions therefore do not preserve arbitrary input dtypes. Signal inputs
use NumPy truth-value conversion.

## Install and run

Clone the repository and create the pinned pixi environment:

```bash
git clone https://github.com/lee101/mojo-vectorbt.git
cd mojo-vectorbt
pixi install
pixi run build
```

`pixi run build` creates `dist/libmojo-vectorbt.so`. Importing
`mojo_vectorbt` also rebuilds the library if `src/kernels.mojo` is newer. A
prebuilt library can be selected with `MOJO_VECTORBT_LIB=/path/to/library.so`.
The usage example above can then be run with `pixi run python`.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz
(72 logical cores), Linux x86-64, Python 3.13.14, NumPy 2.4.6, and vectorbt
1.1.0. Times are the best of three warmed runs and include output allocation
in both implementations.

| case | mojo-vectorbt | vectorbt | result |
| --- | ---: | ---: | ---: |
| `rolling_mean_nb` (1M x 8, window 128) | 257.59 ms | 412.65 ms | 1.60x faster |
| `rolling_std_nb` (1M x 8, window 128) | 173.55 ms | 400.03 ms | 2.31x faster |
| `rolling_max_nb` (500k x 8, window 128) | 134.73 ms | 1205.36 ms | 8.95x faster |
| `returns_nb` (2M x 4) | 94.73 ms | 122.89 ms | 1.30x faster |
| `drawdown_nb` (2M x 4) | 140.91 ms | 264.35 ms | 1.88x faster |
| `sharpe_ratio_nb` (4M x 4) | 32.84 ms | 211.74 ms | 6.45x faster |
| `value_at_risk_nb` (500k x 4) | 28.59 ms | 61.69 ms | 2.16x faster |
| `clean_enex_nb` (2M x 8) | 146.56 ms | 199.87 ms | 1.36x faster |

The largest win is algorithmic: rolling min/max uses a monotonic deque and is
linear in the number of values, while vectorbt 1.1.0 scans each full window.
Sharpe ratio reduces contiguous columns with native-width SIMD and handles
remaining columns with the scalar kernel. Quantile metrics use linear-time
three-way selection instead of fully sorting each column; large matrices
process independent columns in parallel, while small inputs stay serial.

There is no GPU path. These kernels have low arithmetic intensity: Sharpe is a
two-pass memory-bound reduction, and quantile selection is branch-heavy with
irregular memory access. Host/device transfer and launch costs outweigh useful
GPU work for this covered subset.

## How it works

All kernels live in one Mojo compilation unit to avoid repeated fixed compiler
startup cost. Python calls the shared library through `ctypes`, once per array
operation. Arrays are C-contiguous, row-major `float64`; a 2-D value at time
`i`, column `j` is at `i * n_columns + j`, matching vectorbt's processing along
axis 0.

Buffers cross the C ABI as integer addresses. Each exported Mojo function uses
`@export("name")`, `abi("C")`, and reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` internally. Python owns inputs,
outputs, monotonic-deque indices, and quantile scratch buffers. Mojo does not
allocate or retain memory across calls.

## License

MIT
