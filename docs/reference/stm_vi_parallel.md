# Variational inference for supervised LDA (single continuous response).

The model combines unsupervised topic modeling (LDA) with a Gaussian
response on document-level topic proportions.

## Usage

``` r
stm_vi_parallel(
  mod,
  docs,
  y,
  ndsum,
  NZ,
  V,
  K,
  alpha,
  beta,
  update_sigma = TRUE,
  tau = 20L,
  show_progress = TRUE,
  chunk = 5000L
)
```

## Arguments

- mod:

  A list containing the current model state:

  nd

  :   D x K matrix of document-topic counts.

  nw

  :   K x V matrix of topic-word counts.

  eta

  :   Numeric vector of length K; regression coefficients.

  sigma2

  :   Scalar noise variance for the Gaussian response.

- docs:

  IntegerMatrix of size NZ x 3, where each row is a triple (d, v, c) in
  0-based indexing: document index d, word index v, and count c = n_dv.
  Rows with d outside \[0, D-1\] are ignored.

- y:

  NumericVector of length D; response y_d for each document.

- ndsum:

  IntegerVector of length D; total token count per document (that is,
  ndsum\[d\] = sum_v n_dv).

- NZ:

  Integer, number of non-zero entries in docs (rows of docs).

- V:

  Integer, vocabulary size.

- K:

  Integer, number of topics.

- alpha:

  Scalar Dirichlet prior parameter for document-topic distributions
  theta_d (symmetric prior with parameter alpha).

- beta:

  Scalar Dirichlet prior parameter for topic-word distributions phi_k
  (symmetric prior with parameter beta).

- update_sigma:

  Logical; if TRUE, update the noise variance sigma2 from residuals
  y_d - zbar_d^T eta, otherwise keep sigma2 fixed.

- tau:

  Numeric, log-space cutoff used to prune very small topic
  responsibilities phi\[d,i,k\] for numerical stability and efficiency.

- show_progress:

  Logical; if TRUE, print simple progress output during the E-step over
  documents.

- chunk:

  Integer, number of documents to process per parallel block in the
  E-step. Larger values reduce overhead but may use more memory.

## Value

A list with updated variational parameters and diagnostics:

- nd:

  Updated D x K document-topic counts.

- nw:

  Updated K x V topic-word counts.

- eta:

  Updated K-dimensional regression coefficient vector.

- sigma2:

  Updated scalar noise variance.

- elbo:

  Scalar evidence lower bound (approximate).

- label_loglik:

  Gaussian response log-likelihood component.

## Details

\$\$y_d \sim N(zbar_d^T eta, sigma^2).\$\$

This function performs one variational inference sweep with a parallel
document-level E-step and simple updates for the regression parameters.
