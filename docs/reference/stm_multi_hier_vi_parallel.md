# Variational inference for multi-output supervised LDA with hierarchical prior (parallel E-step over documents).

The model is: - Standard LDA for documents: θ_d ~ Dir(α), φ_k ~
Dir(β). - Supervised Gaussian layer: y_d,j \| z̄\_d, η_j, σ_j^2 ~
N(z̄\_d^T η_j, σ_j^2). - Hierarchical prior on regression coefficients:
η_j ~ N(μ, Λ^-1), Λ ~ IW(upsilon, Ω).

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

  List with model state: - nd (D × K) document-topic counts - nw (K × V)
  topic-word counts - eta (K × J) regression coefficients η_j -
  sigma2 (J) per-output noise variance σ_j^2

- docs:

  IntegerMatrix (NZ × 3) with 0-based triples (doc_id, word_id, count).

- y:

  NumericMatrix (D × J) response matrix (NA to skip y_d,j).

- ndsum:

  IntegerVector (D) total token count per document.

- NZ, V, K, J:

  Model sizes (#nonzeros, vocabulary size, topics, responses).

- alpha, beta:

  Dirichlet hyperparameters for θ and φ.

- mu:

  NumericVector (K) prior mean μ for η_j.

- upsilon:

  double, degrees of freedom for inverse-Wishart prior on Λ.

- Omega:

  NumericMatrix (K × K) prior scale matrix for inverse-Wishart.

- update_sigma:

  Logical: update σ_j^2 (true) or keep fixed (false).

- tau:

  Numeric log-cutoff used to prune small φ entries for stability/speed.

- exact_second_moment:

  Logical: if true, use exact E\[z̄ z̄ᵀ\]; if false, use X Xᵀ
  approximation.

- show_progress:

  Logical: print progress information during E-step.

- chunk:

  Number of documents processed per parallel chunk.

## Value

List with updated variational parameters and diagnostics: - nd (D × K)
updated document-topic counts - nw (K × V) updated topic-word counts -
eta (K × J) posterior mean E_q\[η_j\] - eta_se (K × J) approximate
standard errors sqrt(diag Var_q\[η_j\]) - sigma2 (J) updated noise
variances - Lambda_E (K × K) expected precision E_q\[Λ\] -
IW_upsilon_hat scalar posterior dof for inverse-Wishart on Λ -
IW_Omega_hat (K × K) posterior scale matrix for inverse-Wishart on Λ -
elbo scalar approximate evidence lower bound - label_loglik scalar
contribution of supervised likelihood to ELBO
