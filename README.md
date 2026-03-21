# mlstm

**Fast multi-output supervised topic models in R (Rcpp-powered)**

------------------------------------------------------------------------

## Overview

`mlstm` provides efficient implementations of:

-   Latent Dirichlet Allocation (LDA)
-   Supervised Topic Models (STM)
-   Multi-output supervised topic models (MLSTM)

All core algorithms are implemented in C++ via `Rcpp`, enabling fast
estimation even for moderately large datasets.

------------------------------------------------------------------------

## Installation

``` r
# install.packages("remotes")
remotes::install_github("thimeno1993/mlstm")
```

------------------------------------------------------------------------

## Quick Example

### Simulated corpus

``` r
set.seed(123)

D <- 50
V <- 200
K <- 5

NZ_per_doc <- 20
NZ <- D * NZ_per_doc

count <- cbind(
  d = rep(0:(D - 1), each = NZ_per_doc),
  v = sample.int(V, NZ, replace = TRUE) - 1L,
  c = rpois(NZ, 3) + 1
)

Y <- cbind(
  y1 = rnorm(D),
  y2 = rnorm(D)
)
```

------------------------------------------------------------------------

## LDA

``` r
library(mlstm)

mod_lda <- run_lda_gibbs(
  count = count,
  K     = K,
  alpha = 0.1,
  beta  = 0.01,
  n_iter = 20,
  verbose = FALSE
)

str(mod_lda$theta)
str(mod_lda$phi)
```

------------------------------------------------------------------------

## Supervised Topic Model (STM)

``` r
y <- Y[, 1]

set_threads(2)

mod_stm <- run_stm_vi(
  count = count,
  y     = y,
  K     = K,
  alpha = 0.1,
  beta  = 0.01,
  max_iter = 50,
  min_iter = 10,
  verbose  = FALSE
)

y_hat <- ((mod_stm$nd / mod_stm$ndsum) %*% mod_stm$eta)[, 1]
cor(y, y_hat)
```

------------------------------------------------------------------------

## Multi-output STM (MLSTM)

``` r
J <- ncol(Y)

mu      <- rep(0, K)
upsilon <- K + 2
Omega   <- diag(K)

mod_mlstm <- run_mlstm_vi(
  count  = count,
  Y      = Y,
  K      = K,
  alpha  = 0.1,
  beta   = 0.01,
  mu     = mu,
  upsilon = upsilon,
  Omega   = Omega,
  max_iter = 50,
  min_iter = 10,
  verbose  = FALSE
)

Y_hat <- ((mod_mlstm$nd / mod_mlstm$ndsum) %*% mod_mlstm$eta)
cor(Y, Y_hat)
```

------------------------------------------------------------------------

## Data Format

Each row of `count` represents one non-zero document-term entry.

| column | description |
|---|---|
| d | document index (0-based) |
| v | word index (0-based) |
| c | token count |

------------------------------------------------------------------------

## Performance

-   Implemented in C++ via `Rcpp`
-   Parallelized with `RcppParallel`
-   Suitable for large-scale text and supervised learning

------------------------------------------------------------------------

## References

-   Blei, Ng, Jordan (2003)
-   McAuliffe & Blei (2008)

------------------------------------------------------------------------

## Author

Tomoya Himeno

------------------------------------------------------------------------

## License

MIT License

------------------------------------------------------------------------

## Development

``` r
devtools::load_all()
devtools::test()
devtools::check()
```

------------------------------------------------------------------------

## Issues

https://github.com/thimeno1993/mlstm/issues
