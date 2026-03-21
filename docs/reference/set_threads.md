# Set threading options for STM/MLSTM computations

This helper configures OpenMP/BLAS threads to ensure reproducible and
stable performance across the low-level C++ routines used by the
package.

## Usage

``` r
set_threads(num_threads = NULL)
```

## Arguments

- num_threads:

  Integer number of threads. If NULL, use (cores - 1).
