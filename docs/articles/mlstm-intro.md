# Introduction to mlstm

## Overview

`mlstm` provides tools for fitting:

- Latent Dirichlet Allocation (LDA)
- Supervised Topic Models (STM)
- Multi-output supervised topic models (MLSTM)

This vignette shows a minimal end-to-end workflow using simulated data.

## Simulated corpus

We generate a small document-term representation in triplet form.  
Each row of `count` is `(d, v, c)` where:

- `d`: document index (0-based)
- `v`: vocabulary index (0-based)
- `c`: token count

``` r
library(mlstm)

D <- 50
V <- 200
K <- 5

NZ_per_doc <- 20
NZ <- D * NZ_per_doc

count <- cbind(
  d = as.integer(rep(0:(D - 1), each = NZ_per_doc)),
  v = as.integer(sample.int(V, NZ, replace = TRUE) - 1L),
  c = as.integer(rpois(NZ, 3) + 1L)
)

Y <- cbind(
  y1 = rnorm(D),
  y2 = rnorm(D)
)

dim(count)
#> [1] 1000    3
```

``` r
head(count)
#>      d   v c
#> [1,] 0 158 3
#> [2,] 0 178 4
#> [3,] 0  13 6
#> [4,] 0 194 4
#> [5,] 0 169 3
#> [6,] 0  49 6
```

``` r
dim(Y)
#> [1] 50  2
```

## LDA

We first fit an unsupervised LDA model.

``` r
mod_lda <- run_lda_gibbs(
  count = count,
  K = K,
  alpha = 0.1,
  beta = 0.01,
  n_iter = 20,
  verbose = FALSE
)

str(mod_lda$theta)
#>  num [1:50, 1:5] 0.00124 0.3283 0.17736 0.0816 0.02819 ...
```

``` r
str(mod_lda$phi)
#>  num [1:200, 1:5] 1.22e-05 1.22e-05 1.22e-05 2.69e-02 1.22e-05 ...
```

The output typically includes:

- `theta`: document-topic proportions
- `phi`: topic-word distributions
- additional trace information depending on the implementation

## STM

Next, we fit a supervised topic model using a single response variable.

``` r
y <- Y[, 1]

set_threads(2)

mod_stm <- run_stm_vi(
  count = count,
  y = y,
  K = K,
  alpha = 0.1,
  beta = 0.01,
  max_iter = 50,
  min_iter = 10,
  verbose = FALSE
)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
#> [E-step] 50 / 50 (100.0%)
```

``` r

y_hat <- ((mod_stm$nd / mod_stm$ndsum) %*% mod_stm$eta)[, 1]
cor(y, y_hat)
#> [1] 0.8114227
```

If available in the returned object, you can also inspect optimization
traces such as ELBO:

``` r
plot(mod_stm$elbo_trace, type = "l")
plot(mod_stm$label_loglik_trace, type = "l")
```

## MLSTM

Finally, we fit a multi-output supervised topic model.

``` r
mu <- rep(0, K)
upsilon <- K + 2
Omega <- diag(K)

mod_mlstm <- run_mlstm_vi(
  count = count,
  Y = Y,
  K = K,
  alpha = 0.1,
  beta = 0.01,
  mu = mu,
  upsilon = upsilon,
  Omega = Omega,
  max_iter = 50,
  min_iter = 10,
  verbose = FALSE
)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
#> [E-step] 50/50 (100.0%)
```

``` r

Y_hat <- ((mod_mlstm$nd / mod_mlstm$ndsum) %*% mod_mlstm$eta)
cor(Y, Y_hat)
#>         [,1]      [,2]
#> y1 0.2890345 0.1371807
#> y2 0.1087966 0.2273296
```

As with STM, you can inspect fitting diagnostics if stored in the
returned object.

``` r
plot(mod_mlstm$elbo_trace, type = "l")
plot(mod_mlstm$label_loglik_trace, type = "l")
```

## Notes

For package checks and documentation builds, it is better to keep
examples and vignettes light:

- use small synthetic datasets
- keep the number of iterations modest
- avoid verbose console output

This makes the vignette suitable for local builds, GitHub, and CRAN
workflows.

## Session info

``` r
sessionInfo()
#> R version 4.4.1 (2024-06-14 ucrt)
#> Platform: x86_64-w64-mingw32/x64
#> Running under: Windows 11 x64 (build 26200)
#> 
#> Matrix products: default
#> 
#> 
#> locale:
#> [1] LC_COLLATE=Japanese_Japan.utf8  LC_CTYPE=Japanese_Japan.utf8   
#> [3] LC_MONETARY=Japanese_Japan.utf8 LC_NUMERIC=C                   
#> [5] LC_TIME=Japanese_Japan.utf8    
#> 
#> time zone: Etc/GMT-9
#> tzcode source: internal
#> 
#> attached base packages:
#> [1] stats     graphics  grDevices utils     datasets  methods   base     
#> 
#> other attached packages:
#> [1] mlstm_0.1.0
#> 
#> loaded via a namespace (and not attached):
#>  [1] cli_3.6.5           knitr_1.47          rlang_1.1.6        
#>  [4] xfun_0.44           textshaping_0.4.0   jsonlite_2.0.0     
#>  [7] data.table_1.15.4   RcppParallel_5.1.10 htmltools_0.5.8.1  
#> [10] ragg_1.5.0          sass_0.4.9          rmarkdown_2.27     
#> [13] grid_4.4.1          evaluate_1.0.5      jquerylib_0.1.4    
#> [16] fastmap_1.2.0       yaml_2.3.11         lifecycle_1.0.4    
#> [19] compiler_4.4.1      fs_1.6.6            htmlwidgets_1.6.4  
#> [22] Rcpp_1.0.12         rstudioapi_0.16.0   lattice_0.22-6     
#> [25] systemfonts_1.1.0   digest_0.6.35       R6_2.6.1           
#> [28] bslib_0.7.0         Matrix_1.7-0        tools_4.4.1        
#> [31] pkgdown_2.2.0       cachem_1.1.0        desc_1.4.3
```
