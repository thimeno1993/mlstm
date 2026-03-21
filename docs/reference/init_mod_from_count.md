# Initialize LDA/STM state from a (d, v, c) sparse count matrix.

Given a document–term matrix in triplet form (d, v, c) using 0-based
indices, this function initializes the LDA state: - samples initial
topic assignments z, - constructs document–topic counts nd, - constructs
topic–word counts nw, - computes ndsum, nwsum, and normalized topic
proportions X.

## Usage

``` r
init_mod_from_count(count, K = NULL, phi = NULL, seed = NULL)
```

## Arguments

- count:

  Integer matrix with 3 columns representing triples (d, v, c), where d
  and v are 0-based indices.

- K:

  Integer, number of topics. Required if \`phi\` is NULL. If \`phi\` is
  provided, K is inferred from ncol(phi).

- phi:

  Optional numeric matrix of size V × K specifying per-word topic
  probabilities used only during initialization.

- seed:

  Optional integer random seed.

## Value

A list with components:

- z:

  Integer vector (length NZ) of sampled topics, 0-based.

- nd:

  D×K document–topic count matrix.

- nw:

  K×V topic–word count matrix.

- ndsum:

  Integer vector (length D) with row sums of nd.

- nwsum:

  Integer vector (length K) with row sums of nw.

- X:

  D×K matrix of normalized topic proportions nd / ndsum.

- D:

  Number of documents.

- V:

  Vocabulary size.

- K:

  Number of topics.

- NZ:

  Number of non-zero entries (rows in count).

## Details

If a topic–word probability matrix \`phi\` is provided (V × K), initial
topics are sampled according to phi\[v+1, \]. Otherwise, topics are
sampled uniformly from K topics.
