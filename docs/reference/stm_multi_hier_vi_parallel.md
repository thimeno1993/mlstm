# Variational inference for multi-output supervised topic models with hierarchical prior.

The model includes: - LDA structure: theta_d ~ Dir(alpha), phi_k ~
Dir(beta) - Gaussian response: y\[d,j\] ~ N(zbar_d^T eta_j, sigma_j^2) -
Hierarchical prior: eta_j ~ N(mu, Lambda^-1) Lambda ~
inverse-Wishart(upsilon, Omega)

## Usage

``` r
stm_multi_hier_vi_parallel(
  mod,
  docs,
  y,
  ndsum,
  NZ,
  V,
  K,
  J,
  alpha,
  beta,
  mu,
  upsilon,
  Omega,
  update_sigma = TRUE,
  tau = 20L,
  exact_second_moment = FALSE,
  show_progress = TRUE,
  chunk = 5000L
)
```

## Arguments

- mod:

  List with model state: - nd (D x K) document-topic counts - nw (K x V)
  topic-word counts - eta (K x J) regression coefficients - sigma2 (J)
  noise variances

- docs:

  IntegerMatrix (NZ x 3) with (doc_id, word_id, count).

- y:

  NumericMatrix (D x J) response matrix.

- ndsum:

  IntegerVector (D) document token counts.

- NZ, V, K, J:

  Model dimensions.

- alpha, beta:

  Dirichlet hyperparameters.

- mu:

  NumericVector (K) prior mean.

- upsilon:

  Degrees of freedom for inverse-Wishart.

- Omega:

  Scale matrix for inverse-Wishart.

- update_sigma:

  Logical; update sigma2 or not.

- tau:

  Numeric cutoff for stability.

- exact_second_moment:

  Logical flag (currently not used).

- show_progress:

  Logical; print progress.

- chunk:

  Integer; documents per parallel block.

## Value

A list with updated variational parameters and diagnostics:

- nd:

  D x K integer matrix of document-topic counts.

- nw:

  K x V integer matrix of topic-word counts.

- eta:

  K x J numeric matrix of regression coefficients.

- sigma2:

  Length-J numeric vector of noise variances.

- Lambda_E:

  K x K numeric matrix, posterior mean of precision matrix Lambda.

- IW_upsilon_hat:

  Numeric scalar, posterior degrees of freedom.

- IW_Omega_hat:

  K x K numeric matrix, posterior scale matrix.

- elbo:

  Numeric scalar, evidence lower bound.

- label_loglik:

  Numeric scalar, supervised log-likelihood term.
