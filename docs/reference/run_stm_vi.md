# Supervised topic model (STM) variational inference with ELBO-based convergence.

This function performs supervised topic model (STM) using variational
inference. It initializes topic assignments from `count` (optionally
using a topic-word prior `phi`), estimates regression parameters, and
repeatedly calls the C++ routine
[`stm_vi_parallel()`](https://thimeno1993.github.io/mlstm/reference/stm_vi_parallel.md)
until convergence.

## Usage

``` r
run_stm_vi(
  count,
  y,
  K,
  alpha,
  beta,
  phi = NULL,
  seed = NULL,
  max_iter = 200L,
  min_iter = 50L,
  tol_elbo = 1e-04,
  update_sigma = TRUE,
  tau = 20L,
  show_progress = TRUE,
  chunk = 5000L,
  verbose = TRUE,
  sigma2_init = NULL
)
```

## Arguments

- count:

  Integer matrix with 3 columns (d, v, c) in 0-based indexing. Each row
  represents document index `d`, word index `v`, and token count `c`.

- y:

  Numeric vector of length D. Must not contain NA values.

- K:

  Integer, number of topics. Required if `phi` is NULL; ignored if `phi`
  is provided (then `K = ncol(phi)`).

- alpha:

  Dirichlet prior parameter for document-topic distributions.

- beta:

  Dirichlet prior parameter for topic-word distributions.

- phi:

  Optional V x K topic-word probability matrix used only for
  initializing topic assignments.

- seed:

  Optional integer random seed used in the initialization step.

- max_iter:

  Maximum number of variational sweeps.

- min_iter:

  Minimum number of sweeps before checking ELBO convergence.

- tol_elbo:

  Numeric tolerance for relative ELBO change.

- update_sigma:

  Logical; if TRUE, update `sigma2` each sweep.

- tau:

  Numeric log-space cutoff used in
  [`stm_vi_parallel()`](https://thimeno1993.github.io/mlstm/reference/stm_vi_parallel.md).

- show_progress:

  Logical; print low-level progress inside C++.

- chunk:

  Integer; number of documents per parallel block.

- verbose:

  Logical; print ELBO and relative change per sweep.

- sigma2_init:

  Optional numeric scalar specifying the initial noise variance. If
  NULL, `sigma2` is estimated once by least squares.

## Value

A list containing:

- nd:

  D x K document-topic count matrix.

- nw:

  K x V topic-word count matrix.

- ndsum:

  Length-D vector of document token counts.

- nwsum:

  Length-K vector of topic token counts.

- eta:

  K-dimensional regression coefficient vector.

- sigma2:

  Final noise variance.

- phi:

  V x K topic-word posterior mean.

- theta:

  D x K document-topic posterior mean.

- elbo:

  Final ELBO.

- label_loglik:

  Final supervised term.

- elbo_trace:

  ELBO values per sweep.

- label_loglik_trace:

  Label log-likelihood per sweep.

- n_iter:

  Number of iterations actually performed.

- D, V, K, NZ:

  Model dimensions.

## Details

Convergence is assessed based on the relative changes in the evidence
lower bound (ELBO) and the supervised label log-likelihood: \$\$
\frac{\mathrm{ELBO}\_t -
\mathrm{ELBO}\_{t-1}}{\|\mathrm{ELBO}\_{t-1}\|}, \qquad \frac{\ell_t -
\ell\_{t-1}}{\|\ell\_{t-1}\|}. \$\$ After a minimum number of
iterations, the algorithm is declared to have converged when both
quantities are non-negative and smaller than the prescribed tolerance.

\*\*Important:\*\* This function assumes that the response vector `y`
contains \*\*no NA\*\* values. The underlying C++ implementation does
not skip missing responses and requires `y[d]` to be finite for all
documents.
