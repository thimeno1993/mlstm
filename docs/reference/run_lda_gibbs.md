# Collapsed LDA Gibbs sampling for sparse (d, v, c) triplet data.

This function performs collapsed Gibbs sampling for the standard LDA
model using a sparse document–term representation:

1.  initializes the LDA state via
    [`init_mod_from_count()`](https://thimeno1993.github.io/mlstm/reference/init_mod_from_count.md),

2.  runs `n_iter` iterations of the C++ Gibbs kernel
    [`eLDA_pass_b_fast()`](https://thimeno1993.github.io/mlstm/reference/eLDA_pass_b_fast.md),

3.  returns the final model state, including posterior topic–word and
    document–topic distributions.

## Usage

``` r
run_lda_gibbs(
  count,
  K,
  alpha,
  beta,
  n_iter = 100L,
  phi = NULL,
  seed = NULL,
  verbose = TRUE,
  progress_every = 10L
)
```

## Arguments

- count:

  Integer matrix of size NZ × 3 with rows (d, v, c) in 0-based indexing:
  document index `d`, word index `v`, and count `c` for that pair.

- K:

  Integer, number of topics. Required unless `phi` is supplied. If `phi`
  is provided, `K` is inferred from `ncol(phi)`.

- alpha:

  Scalar Dirichlet prior parameter for document–topic distributions.

- beta:

  Scalar Dirichlet prior parameter for topic–word distributions.

- n_iter:

  Integer, number of Gibbs sweeps to run.

- phi:

  Optional V × K topic–word probability matrix used only for
  initializing topic assignments in
  [`init_mod_from_count()`](https://thimeno1993.github.io/mlstm/reference/init_mod_from_count.md).

- seed:

  Optional integer random seed passed to the initializer.

- verbose:

  Logical; if `TRUE`, print progress messages.

- progress_every:

  Integer; print progress every this many iterations.

## Value

A list `mod` containing:

- z:

  Integer vector of length NZ; final topic assignments (0-based).

- nd:

  D × K document–topic count matrix.

- nw:

  K × V topic–word count matrix.

- ndsum:

  Integer vector of length D; document token counts.

- nwsum:

  Integer vector of length K; topic token counts.

- phi:

  V × K topic–word posterior mean \\p(w \mid z=k)\\ computed from `nw`.

- theta:

  D × K document–topic posterior mean \\p(z=k \mid d)\\ computed from
  `nd`.

- loglik_trace:

  Vector of log-likelihoods.

- D:

  Number of documents.

- V:

  Vocabulary size.

- K:

  Number of topics.

- NZ:

  Number of non-zero (d, v, c) entries.
