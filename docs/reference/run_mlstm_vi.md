# Multi-level supervised topic model (MLSTM) via variational inference.

This function fits a multi-output supervised LDA model with a
hierarchical prior on regression coefficients: \$\$\eta_j \sim N(\mu,
\Lambda^{-1}), \quad \Lambda \sim \text{IW}(\upsilon, \Omega).\$\$

## Usage

``` r
run_mlstm_vi(
  count,
  Y,
  K,
  alpha,
  beta,
  mu,
  upsilon,
  Omega,
  phi = NULL,
  seed = NULL,
  max_iter = 200L,
  min_iter = 50L,
  tol_elbo = 1e-04,
  update_sigma = TRUE,
  tau = 20L,
  exact_second_moment = FALSE,
  show_progress = TRUE,
  chunk = 5000L,
  verbose = TRUE,
  sigma2_init = NULL
)
```

## Arguments

- count:

  Integer matrix with 3 columns (d, v, c), using 0-based indices. Each
  row represents document index `d`, word index `v`, and token count
  `c`.

- Y:

  Numeric matrix of size D x J containing J response variables for each
  of the D documents. NA values are allowed and are ignored in the
  initial regression used to seed `eta` and `sigma2`.

- K:

  Integer, number of topics. Required if `phi` is `NULL`; ignored if
  `phi` is supplied, in which case `K = ncol(phi)`.

- alpha:

  Dirichlet prior parameter for document-topic distributions.

- beta:

  Dirichlet prior parameter for topic-word distributions.

- mu:

  Numeric vector of length K; prior mean for each \\\eta_j\\.

- upsilon:

  Scalar degrees of freedom for the inverse-Wishart prior on the
  precision matrix \\\Lambda\\.

- Omega:

  Numeric K x K positive-definite scale matrix for the inverse-Wishart
  prior.

- phi:

  Optional numeric matrix of size V x K used only to initialize topic
  assignments via
  [`init_mod_from_count()`](https://thimeno1993.github.io/mlstm/reference/init_mod_from_count.md).

- seed:

  Optional integer random seed used for initialization.

- max_iter:

  Maximum number of variational sweeps.

- min_iter:

  Minimum number of sweeps before checking convergence.

- tol_elbo:

  Numeric tolerance for the relative ELBO change used in the convergence
  criterion.

- update_sigma:

  Logical; if TRUE, update `sigma2` inside
  [`stm_multi_hier_vi_parallel()`](https://thimeno1993.github.io/mlstm/reference/stm_multi_hier_vi_parallel.md).
  If FALSE, keep `sigma2` fixed at its initialized value.

- tau:

  Log-space cutoff for local topic responsibilities in the C++ routine
  (controls pruning for stability and speed).

- exact_second_moment:

  Logical; reserved flag intended to control whether the exact second
  moment \\E\[\bar{z}\bar{z}^\top\]\\ is accumulated in the E-step.
  \*\*Currently this option has no effect\*\*: the underlying C++
  implementation ignores the accumulated second-moment matrix when
  updating the variational parameters, and only an approximate moment
  based on \\\bar{z}\bar{z}^\top\\ is effectively used.

- show_progress:

  Logical; forwarded to
  [`stm_multi_hier_vi_parallel()`](https://thimeno1993.github.io/mlstm/reference/stm_multi_hier_vi_parallel.md).

- chunk:

  Integer; number of documents per parallel block in the C++ E-step.

- verbose:

  Logical; if TRUE, print ELBO and its relative change at each sweep.

- sigma2_init:

  Optional numeric scalar or length-J vector specifying the initial
  noise variances. If `NULL`, `sigma2` is estimated for each response
  dimension by least squares regression of `Y[, j]` on initial topic
  proportions.

## Value

A list `mod` containing (at least):

- nd:

  D x K document-topic counts.

- nw:

  K x V topic-word counts.

- ndsum:

  Integer vector of length D; document token counts.

- nwsum:

  Integer vector of length K; topic token counts.

- eta:

  K x J matrix of regression coefficients.

- sigma2:

  Length-J vector of noise variances.

- Lambda_E:

  K x K posterior mean of \\\Lambda\\ (if returned by C++).

- IW_upsilon_hat:

  Posterior degrees of freedom (if returned by C++).

- IW_Omega_hat:

  Posterior scale matrix (if returned by C++).

- phi:

  V x K topic-word posterior mean \\p(w \mid z=k)\\ computed from `nw`.

- theta:

  D x K document-topic posterior mean \\p(z=k \mid d)\\ computed from
  `nd`.

- elbo:

  Final ELBO value.

- label_loglik:

  Final label log-likelihood term.

- elbo_trace:

  Numeric vector of ELBO values over iterations.

- label_loglik_trace:

  Numeric vector of label log-likelihoods.

- n_iter:

  Number of sweeps actually performed.

- D:

  Number of documents.

- V:

  Vocabulary size.

- K:

  Number of topics.

- J:

  Number of response dimensions.

- NZ:

  Number of non-zero (d, v, c) entries.

## Details

The latent topic layer is standard LDA, and each response dimension j
follows a Gaussian regression on document-level topic proportions.
Variational inference is performed by repeated calls to the C++ routine
[`stm_multi_hier_vi_parallel()`](https://thimeno1993.github.io/mlstm/reference/stm_multi_hier_vi_parallel.md)
until convergence or a maximum number of sweeps is reached.

Convergence is assessed based on the relative changes in the evidence
lower bound (ELBO) and the supervised label log-likelihood: \$\$
\frac{\mathrm{ELBO}\_t -
\mathrm{ELBO}\_{t-1}}{\|\mathrm{ELBO}\_{t-1}\|}, \qquad \frac{\ell_t -
\ell\_{t-1}}{\|\ell\_{t-1}\|}. \$\$ After a minimum number of
iterations, the algorithm is declared to have converged when both
quantities are non-negative and smaller than the prescribed tolerance.
