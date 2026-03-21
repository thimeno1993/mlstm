# One Gibbs sampling sweep for LDA (collapsed) using document–term list.

This function performs a single collapsed Gibbs sampling pass over all
non-zero document–term entries. Each (d, v, count) triple is treated as
\`count\` replicated word tokens sharing the same topic assignment.

## Usage

``` r
eLDA_pass_b_fast(mod, count, ndsum, NZ, V, K, alpha, beta)
```

## Arguments

- mod:

  List with current sampler state: `z`, `nd`, `nw`, and `nwsum` as
  described above.

- count:

  IntegerMatrix of size NZ×3, where each row is a triple (d, v, c) with
  0-based indices: document index `d`, word index `v`, and count `c` for
  that (doc, word) pair.

- ndsum:

  IntegerVector of length D; total token count per document (i.e.,
  `ndsum[d] = sum_k nd(d,k)`). Updated in place.

- NZ:

  Integer, number of non-zero entries (rows in `count` and length of
  `z`).

- V:

  Integer, vocabulary size.

- K:

  Integer, number of topics.

- alpha:

  Scalar Dirichlet prior parameter for document–topic distributions
  \\\theta_d\\ (symmetric).

- beta:

  Scalar Dirichlet prior parameter for topic–word distributions
  \\\phi_k\\ (symmetric).

## Value

A list with updated state:

- z:

  Updated topic assignment vector (length NZ).

- nd:

  Updated D×K document–topic counts.

- nw:

  Updated K×V topic–word counts.

- nwsum:

  Updated total word counts per topic.

## Details

The state is stored in a list \`mod\` containing:

- z:

  Integer vector of length NZ; topic assignment for each (d, v, count)
  triple.

- nd:

  D×K integer matrix; document–topic counts.

- nw:

  K×V integer matrix; topic–word counts.

- nwsum:

  Integer vector of length K; total word count per topic.
