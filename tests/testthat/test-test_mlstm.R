test_that("test fit LDA/STM/MLSTM", {

  # prepare simulated corpus ------------------------------------------------

  D <- 1000        # number of documents
  V <- 5000        # vocabulary size
  NZ_per_doc <- 100 # average non-zero entries per document
  lambda <- 3       # mean count per entry (Poisson)
  K     <- 10L    # number of topics
  alpha <- 0.1
  beta  <- 0.01

  # total NZ
  NZ <- D * NZ_per_doc

  # document index (0-based)
  d <- rep(0:(D - 1L), each = NZ_per_doc)

  # random word indices (0-based)
  v <- sample.int(V, size = NZ, replace = TRUE) - 1L

  # Poisson counts (>=1)
  c <- rpois(NZ, lambda) + 1

  # corpus
  count <- cbind(
    d = as.integer(d),
    v = as.integer(v),
    c = as.integer(c)
  )

  # surpervised data
  Y <- cbind(
    a = rnorm(D),
    b = rnorm(D),
    c = rnorm(D)
  )


  # LDA ---------------------------------------------------------------------

  mod_lda <- run_lda_gibbs(
    count = count,
    K     = K,
    alpha = alpha,
    beta  = beta,
    n_iter = 100,
    verbose = TRUE
  )

  str(mod_lda$phi)
  str(mod_lda$theta)
  plot(mod_lda$loglik_trace)


  # STM --------------------------------------------------------------------

  y <- Y[,1]

  set_threads()

  mod_stm <- run_stm_vi(
    count = count,
    y     = y,
    K     = K,
    alpha = alpha,
    beta  = beta,
    max_iter = 200,
    min_iter = 20,
    verbose  = TRUE,
    update_sigma  = FALSE,
    sigma2_init = .5^2,
    chunk = 10
  )

  plot(mod_stm$elbo_trace, type = "l")
  plot(mod_stm$label_loglik_trace, type = "l")

  y_hat = ((mod_stm$nd / mod_stm$ndsum) %*% mod_stm$eta)[, 1]
  cor(y, y_hat)


  # MLSTM ------------------------------------------------------------------

  J <- ncol(Y)

  set_threads()

  mu      <- rep(0, K)
  upsilon <- K + 2
  Omega   <- diag(K)

  mod_mlstm <- run_mlstm_vi(
    count  = count,
    Y      = Y,
    K      = K,
    alpha  = alpha,
    beta   = beta,
    mu     = mu,
    upsilon = upsilon,
    Omega   = Omega,
    max_iter = 200,
    min_iter = 20,
    verbose  = TRUE
  )

  plot(mod_mlstm$elbo_trace, type = "l")
  plot(mod_mlstm$label_loglik_trace, type = "l")

  Y_hat = ((mod_mlstm$nd / mod_mlstm$ndsum) %*% mod_mlstm$eta)
  cor(Y, Y_hat)

})
