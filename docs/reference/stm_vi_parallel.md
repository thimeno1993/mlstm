# Variational inference for supervised LDA (single continuous response).

The model combines unsupervised topic modeling (LDA) with a Gaussian
response on the document-level topic proportions z̄\_d: \$\$y_d \mid
zbar_d, eta \sigma^2 \sim N(zbar_d^T eta\\ \sigma^2).\$\$

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

  :   D×K matrix of document–topic counts.

  nw

  :   K×V matrix of topic–word counts.

  eta

  :   Numeric vector of length K; regression coefficients.

  sigma2

  :   Scalar noise variance for the Gaussian response.

- docs:

  IntegerMatrix of size NZ×3, where each row is a triple (d, v, c) in
  0-based indexing: document index d, word index v, and count c = n_d,v.
  Rows with d outside \[0, D-1\] are ignored.

- y:

  NumericVector of length D; response y_d for each document.

- ndsum:

  IntegerVector of length D; total token count per document (i.e.,
  ndsum\[d\] = sum_v n_d,v).

- NZ:

  Integer, number of non-zero entries in docs (rows of docs).

- V:

  Integer, vocabulary size.

- K:

  Integer, number of topics.

- alpha:

  Scalar Dirichlet prior parameter for document–topic distributions θ_d
  (symmetric prior with parameter α).

- beta:

  Scalar Dirichlet prior parameter for topic–word distributions β_k
  (symmetric prior with parameter β).

- update_sigma:

  Logical; if TRUE, update the noise variance σ² from residuals y_d -
  zbar_d^T eta otherwise keep σ² fixed.

- tau:

  Numeric, log-space cutoff used to prune very small topic
  responsibilities φ_d,i,k for numerical stability and efficiency.

- show_progress:

  Logical; if TRUE, print simple progress output during the E-step over
  documents.

- chunk:

  Integer, number of documents to process per parallel block in the
  E-step. Larger values reduce overhead but may use more memory.

## Value

A list with updated variational parameters and diagnostics:

- nd:

  Updated D×K document–topic counts.

- nw:

  Updated K×V topic–word counts.

- eta:

  Updated K-dimensional regression coefficient vector.

- sigma2:

  Updated scalar noise variance.

- elbo:

  Scalar evidence lower bound (approximate).

- label_loglik:

  Gaussian response log-likelihood component.

## Details

This function performs one variational inference sweep with a parallel
document-level E-step and simple updates for the regression parameters.
