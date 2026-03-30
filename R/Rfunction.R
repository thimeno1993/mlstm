#' Set threading options for STM/MLSTM computations
#'
#' This helper configures OpenMP/BLAS threads to ensure reproducible and
#' stable performance across the low-level C++ routines used by the package.
#'
#' @param num_threads Integer number of threads. If NULL, use (cores - 1).
#'
#' @return Invisibly returns an integer giving the number of threads used.
#'
#' @export
set_threads <- function(num_threads = NULL) {
  if (is.null(num_threads)) {
    num_threads <- max(1L, parallel::detectCores() - 1L)
  }
  Sys.setenv(OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
  RcppParallel::setThreadOptions(numThreads = num_threads)
  invisible(num_threads)
}


#' Initialize LDA/STM state from a (d, v, c) sparse count matrix.
#'
#' Given a document-term matrix in triplet form (d, v, c) using 0-based indices,
#' this function initializes the LDA state:
#' - samples initial topic assignments z,
#' - constructs document-topic counts nd,
#' - constructs topic-word counts nw,
#' - computes ndsum, nwsum, and normalized topic proportions X.
#'
#' If a topic-word probability matrix `phi` is provided (V x K),
#' initial topics are sampled according to phi[v+1, ].
#' Otherwise, topics are sampled uniformly from K topics.
#'
#' @param count Integer matrix with 3 columns representing triples (d, v, c),
#'   where d and v are 0-based indices.
#' @param K Integer, number of topics. Required if `phi` is NULL.
#'   If `phi` is provided, K is inferred from ncol(phi).
#' @param phi Optional numeric matrix of size V x K specifying per-word topic
#'   probabilities used only during initialization.
#' @param seed Optional integer random seed.
#'
#' @return A list with components:
#'   \describe{
#'     \item{z}{Integer vector (length NZ) of sampled topics, 0-based.}
#'     \item{nd}{DxK document-topic count matrix.}
#'     \item{nw}{KxV topic-word count matrix.}
#'     \item{ndsum}{Integer vector (length D) with row sums of nd.}
#'     \item{nwsum}{Integer vector (length K) with row sums of nw.}
#'     \item{X}{DxK matrix of normalized topic proportions nd / ndsum.}
#'     \item{D}{Number of documents.}
#'     \item{V}{Vocabulary size.}
#'     \item{K}{Number of topics.}
#'     \item{NZ}{Number of non-zero entries (rows in count).}
#'   }
#'
#' @export
init_mod_from_count <- function(count, K = NULL, phi = NULL, seed = NULL) {
  if (!is.matrix(count))
    stop("`count` must be a matrix with columns (d, v, c).")

  if (ncol(count) != 3L)
    stop("`count` must have exactly 3 columns: (d, v, c).")

  if (!is.integer(count)) {
    count <- apply(count, 2L, as.integer)
    count <- as.matrix(count)
  }

  # extract columns (support both named and unnamed matrices)
  cn <- colnames(count)
  d_col <- if (!is.null(cn) && "d" %in% cn) which(cn == "d") else 1L
  v_col <- if (!is.null(cn) && "v" %in% cn) which(cn == "v") else 2L
  c_col <- if (!is.null(cn) && "c" %in% cn) which(cn == "c") else 3L

  D  <- max(count[, d_col]) + 1L
  V  <- max(count[, v_col]) + 1L
  NZ <- nrow(count)

  use_phi <- !is.null(phi)

  if (use_phi) {
    if (!is.matrix(phi))
      stop("`phi` must be a numeric matrix of size V x K.")

    if (nrow(phi) != V)
      stop("nrow(phi) must equal vocabulary size V.")

    if (!is.null(K) && K != ncol(phi)) {
      warning("K is overridden by ncol(phi).")
    }
    K <- ncol(phi)
  } else {
    if (is.null(K))
      stop("K must be provided when phi = NULL.")
    K <- as.integer(K)
  }

  if (!is.null(seed)) set.seed(seed)

  # --- sample initial topics -----------------------------------------
  z <- integer(NZ)

  if (use_phi) {
    v_index_list <- split(seq_len(NZ), count[, v_col])
    for (v0 in seq_len(V) - 1L) {
      idxs <- v_index_list[[as.character(v0)]]
      if (is.null(idxs) || length(idxs) == 0L) next

      p <- phi[v0 + 1L, ]
      if (sum(p) <= 0) p[] <- 1
      p <- p / sum(p)

      z[idxs] <- sample.int(K, length(idxs), replace = TRUE, prob = p) - 1L
    }
  } else {
    z[] <- sample.int(K, NZ, replace = TRUE) - 1L
  }

  # --- count aggregation ----------------------------------------------
  dt <- data.table::data.table(
    d = count[, d_col],
    v = count[, v_col],
    n = count[, c_col],
    z = z
  )

  dz <- stats::aggregate(n ~ d + z, data = as.data.frame(dt), FUN  = sum)
  vz <- stats::aggregate(n ~ v + z, data = as.data.frame(dt), FUN  = sum)

  nd <- Matrix::sparseMatrix(i = dz$d + 1L, j = dz$z + 1L,
                             x = dz$n, dims = c(D, K))
  nd <- as.matrix(nd)
  ndsum <- as.integer(rowSums(nd))

  nw <- Matrix::sparseMatrix(i = vz$z + 1L, j = vz$v + 1L,
                             x = vz$n, dims = c(K, V))
  nw <- as.matrix(nw)
  nwsum <- as.integer(rowSums(nw))

  X <- nd / ndsum

  list(
    z = z, nd = nd, nw = nw,
    ndsum = ndsum, nwsum = nwsum,
    X = X, D = D, V = V, K = K, NZ = NZ
  )
}




#' Collapsed LDA Gibbs sampling for sparse (d, v, c) triplet data.
#'
#' This function performs collapsed Gibbs sampling for the standard LDA model
#' using a sparse document-term representation:
#' \enumerate{
#'   \item initializes the LDA state via \code{init_mod_from_count()},
#'   \item runs \code{n_iter} iterations of the C++ Gibbs kernel
#'         \code{eLDA_pass_b_fast()},
#'   \item returns the final model state, including posterior topic-word
#'         and document-topic distributions.
#' }
#'
#' @param count Integer matrix of size NZ x 3 with rows (d, v, c) in 0-based
#'   indexing: document index \code{d}, word index \code{v}, and count
#'   \code{c} for that pair.
#' @param K Integer, number of topics. Required unless \code{phi} is supplied.
#'   If \code{phi} is provided, \code{K} is inferred from \code{ncol(phi)}.
#' @param alpha Scalar Dirichlet prior parameter for document-topic
#'   distributions.
#' @param beta Scalar Dirichlet prior parameter for topic-word
#'   distributions.
#' @param n_iter Integer, number of Gibbs sweeps to run.
#' @param phi Optional V x K topic-word probability matrix used only for
#'   initializing topic assignments in \code{init_mod_from_count()}.
#' @param seed Optional integer random seed passed to the initializer.
#' @param verbose Logical; if \code{TRUE}, print progress messages.
#' @param progress_every Integer; print progress every this many iterations.
#'
#' @return A list \code{mod} containing:
#'   \describe{
#'     \item{z}{Integer vector of length NZ; final topic assignments (0-based).}
#'     \item{nd}{D x K document-topic count matrix.}
#'     \item{nw}{K x V topic-word count matrix.}
#'     \item{ndsum}{Integer vector of length D; document token counts.}
#'     \item{nwsum}{Integer vector of length K; topic token counts.}
#'     \item{phi}{V x K topic-word posterior mean
#'       \eqn{p(w \mid z=k)} computed from \code{nw}.}
#'     \item{theta}{D x K document-topic posterior mean
#'       \eqn{p(z=k \mid d)} computed from \code{nd}.}
#'     \item{loglik_trace}{Vector of log-likelihoods.}
#'     \item{D}{Number of documents.}
#'     \item{V}{Vocabulary size.}
#'     \item{K}{Number of topics.}
#'     \item{NZ}{Number of non-zero (d, v, c) entries.}
#'   }
#'
#' @export
run_lda_gibbs <- function(count,
                          K,
                          alpha,
                          beta,
                          n_iter = 100L,
                          phi = NULL,
                          seed = NULL,
                          verbose = TRUE,
                          progress_every = 10L) {

  # --- 1) initialize state -----------------------------------------------
  init <- init_mod_from_count(count, K = K, phi = phi, seed = seed)

  mod <- list(
    z     = init$z,
    nd    = init$nd,
    nw    = init$nw,
    nwsum = init$nwsum
  )
  ndsum <- init$ndsum

  NZ <- init$NZ
  V  <- init$V
  K  <- init$K

  n_iter <- as.integer(n_iter)
  if (n_iter <= 0L) stop("`n_iter` must be positive.")

  # ensure `count` is integer for the C++ kernel
  if (!is.integer(count)) {
    count <- apply(count, 2L, as.integer)
    count <- as.matrix(count)
  }

  loglik_trace <- numeric(n_iter)

  # --- 2) Gibbs sweeps ----------------------------------------------------
  for (iter in seq_len(n_iter)) {
    mod <- eLDA_pass_b_fast(
      mod   = mod,
      count = count,
      ndsum = ndsum,
      NZ    = NZ,
      V     = V,
      K     = K,
      alpha = alpha,
      beta  = beta
    )

    loglik_trace[iter] <- mod$log_likelihood

    if (verbose && (iter %% progress_every == 0L || iter == n_iter)) {
      message(sprintf("[run_lda_gibbs] iteration %d / %d", iter, n_iter))
    }
  }

  # --- 3) finalize output -------------------------------------------------
  mod$ndsum <- ndsum
  mod$D     <- init$D
  mod$V     <- V
  mod$K     <- K
  mod$NZ    <- NZ
  mod$loglik_trace <- loglik_trace[seq_len(n_iter)]

  # topic-wise token counts
  nwsum <- as.integer(rowSums(mod$nw))
  mod$nwsum <- nwsum

  # topic-word posterior mean: V x K
  mod$phi <- t((mod$nw + beta) / (nwsum + V * beta))

  # document-topic posterior mean: D x K
  mod$theta <- (mod$nd + alpha) / (mod$ndsum + K * alpha)

  mod
}




#' Supervised topic model (STM) variational inference with ELBO-based convergence.
#'
#' This function performs supervised topic model (STM) using variational inference.
#' It initializes topic assignments from \code{count} (optionally using a
#' topic-word prior \code{phi}), estimates regression parameters, and repeatedly
#' calls the C++ routine \code{stm_vi_parallel()} until convergence.
#'
#' Convergence is assessed based on the relative changes in the evidence lower
#' bound (ELBO) and the supervised label log-likelihood:
#'   \deqn{
#'     \frac{\mathrm{ELBO}_t - \mathrm{ELBO}_{t-1}}{|\mathrm{ELBO}_{t-1}|},
#'     \qquad
#'     \frac{\ell_t - \ell_{t-1}}{|\ell_{t-1}|}.
#'   }
#' After a minimum number of iterations, the algorithm is declared to have
#' converged when both quantities are non-negative and smaller than the
#' prescribed tolerance.
#'
#' **Important:**
#' This function assumes that the response vector \code{y} contains **no NA**
#' values. The underlying C++ implementation does not skip missing responses
#' and requires \code{y[d]} to be finite for all documents.
#'
#' @param count Integer matrix with 3 columns (d, v, c) in 0-based indexing.
#'   Each row represents document index \code{d}, word index \code{v}, and
#'   token count \code{c}.
#' @param y Numeric vector of length D. Must not contain NA values.
#' @param K Integer, number of topics. Required if \code{phi} is NULL;
#'   ignored if \code{phi} is provided (then \code{K = ncol(phi)}).
#' @param alpha Dirichlet prior parameter for document-topic distributions.
#' @param beta Dirichlet prior parameter for topic-word distributions.
#' @param phi Optional V x K topic-word probability matrix used only for
#'   initializing topic assignments.
#' @param seed Optional integer random seed used in the initialization step.
#' @param max_iter Maximum number of variational sweeps.
#' @param min_iter Minimum number of sweeps before checking ELBO convergence.
#' @param tol_elbo Numeric tolerance for relative ELBO change.
#' @param update_sigma Logical; if TRUE, update \code{sigma2} each sweep.
#' @param tau Numeric log-space cutoff used in \code{stm_vi_parallel()}.
#' @param show_progress Logical; print low-level progress inside C++.
#' @param chunk Integer; number of documents per parallel block.
#' @param verbose Logical; print ELBO and relative change per sweep.
#' @param sigma2_init Optional numeric scalar specifying the initial
#'   noise variance. If NULL, \code{sigma2} is estimated once by least squares.
#'
#' @return A list containing:
#'   \describe{
#'     \item{nd}{D x K document-topic count matrix.}
#'     \item{nw}{K x V topic-word count matrix.}
#'     \item{ndsum}{Length-D vector of document token counts.}
#'     \item{nwsum}{Length-K vector of topic token counts.}
#'     \item{eta}{K-dimensional regression coefficient vector.}
#'     \item{sigma2}{Final noise variance.}
#'     \item{phi}{V x K topic-word posterior mean.}
#'     \item{theta}{D x K document-topic posterior mean.}
#'     \item{elbo}{Final ELBO.}
#'     \item{label_loglik}{Final supervised term.}
#'     \item{elbo_trace}{ELBO values per sweep.}
#'     \item{label_loglik_trace}{Label log-likelihood per sweep.}
#'     \item{n_iter}{Number of iterations actually performed.}
#'     \item{D, V, K, NZ}{Model dimensions.}
#'   }
#'
#' @export
run_stm_vi <- function(count,
                        y,
                        K,
                        alpha,
                        beta,
                        phi           = NULL,
                        seed          = NULL,
                        max_iter      = 200L,
                        min_iter      = 50L,
                        tol_elbo      = 1e-4,
                        update_sigma  = TRUE,
                        tau           = 20L,
                        show_progress = TRUE,
                        chunk         = 5000L,
                        verbose       = TRUE,
                        sigma2_init   = NULL) {

  # --- basic checks -------------------------------------------------------
  if (!is.numeric(y))
    stop("`y` must be a numeric vector.")

  if(anyNA(y))
    stop("y must not contain NA.")

  # initialize LDA state
  init <- init_mod_from_count(count, K = K, phi = phi, seed = seed)

  D  <- init$D
  V  <- init$V
  K  <- init$K
  NZ <- init$NZ

  if (length(y) != D)
    stop("length(y) must equal the number of documents D.")

  ndsum <- init$ndsum

  # --- initialize eta and sigma2 -----------------------------------------
  # X0 is used only internally; not returned
  X0 <- init$nd / ndsum

  # initialize eta
  eta_init <- numeric(K)
  ok <- which(!is.na(y))

  if (length(ok) > K) {
    XtX <- crossprod(X0[ok, , drop = FALSE])
    Xty <- crossprod(X0[ok, , drop = FALSE], y[ok])
    eta_init <- solve(XtX + diag(1e-6, K), Xty)
  } else {
    eta_init[] <- 0
  }

  # initialize sigma2
  if (!is.null(sigma2_init)) {
    # --- user supplied ---------------------------------------------
    sigma2 <- as.numeric(sigma2_init)
  } else {
    # --- estimate from residual ------------------------------------
    if (length(ok) > K) {
      r  <- y[ok] - X0[ok, , drop = FALSE] %*% eta_init
      df <- max(1L, length(ok) - K)
      sigma2 <- as.numeric(crossprod(r) / df)
    } else {
      sigma2 <- 1.0
    }
  }

# --- initial mod list --------------------------------------------
mod <- list(
  nd     = init$nd,
  nw     = init$nw,
  eta    = eta_init,
  sigma2 = sigma2     # << user-specified or estimated
)

if (!is.integer(count)) {
  count <- apply(count, 2L, as.integer)
  count <- as.matrix(count)
}
y_vec <- as.numeric(y)

max_iter <- as.integer(max_iter)
min_iter <- as.integer(min_iter)
if (max_iter <= 0L) stop("`max_iter` must be positive.")
if (min_iter <= 0L) min_iter <- 1L

# NEW: use the same tolerance unless you want to introduce tol_label
tol_label <- tol_elbo

elbo_trace         <- numeric(max_iter)
label_loglik_trace <- numeric(max_iter)

prev_elbo   <- NA_real_
prev_label  <- NA_real_
delta_elbo  <- NA_real_
delta_label <- NA_real_

# --- VI sweeps ----------------------------------------------------------
for (iter in seq_len(max_iter)) {

  mod <- stm_vi_parallel(
    mod           = mod,
    docs          = count,
    y             = y_vec,
    ndsum         = ndsum,
    NZ            = NZ,
    V             = V,
    K             = K,
    alpha         = alpha,
    beta          = beta,
    update_sigma  = update_sigma,
    tau           = tau,
    show_progress = show_progress,
    chunk         = chunk
  )

  elbo_trace[iter]         <- mod$elbo
  label_loglik_trace[iter] <- mod$label_loglik

  # relative changes
  if (!is.na(prev_elbo) && prev_elbo != 0) {
    delta_elbo <- (mod$elbo - prev_elbo) / abs(prev_elbo)
  } else {
    delta_elbo <- NA_real_
  }

  if (!is.na(prev_label) && prev_label != 0) {
    delta_label <- (mod$label_loglik - prev_label) / abs(prev_label)
  } else {
    delta_label <- NA_real_
  }

  if (verbose) {
    msg <- sprintf(
      "[run_stm_vi] iter = %d / %d; ELBO = %.5f; dELBO = %s; label = %.5f; dLabel = %s",
      iter, max_iter, mod$elbo,
      if (is.na(delta_elbo)) "NA" else sprintf("%.5e", delta_elbo),
      mod$label_loglik,
      if (is.na(delta_label)) "NA" else sprintf("%.5e", delta_label)
    )
    message(msg)
  }

  # convergence: after min_iter, both relative changes are small and non-negative
  if (iter >= min_iter &&
      !is.na(delta_elbo)  && delta_elbo  >= 0 && delta_elbo  <= tol_elbo &&
      !is.na(delta_label) && delta_label >= 0 && delta_label <= tol_label) {

    if (verbose) {
      message(sprintf(
        "[run_stm_vi] converged at iter %d with dELBO = %.5e and dLabel = %.5e",
        iter, delta_elbo, delta_label
      ))
    }
    break
  }

  prev_elbo  <- mod$elbo
  prev_label <- mod$label_loglik
}

n_done <- iter


  # --- finalize ------------------------------------------------------------
  mod$ndsum <- ndsum
  nwsum     <- as.integer(rowSums(mod$nw))
  mod$nwsum <- nwsum

  mod$phi   <- t((mod$nw + beta) / (nwsum + V * beta))
  mod$theta <- (mod$nd + alpha) / (ndsum + K * alpha)

  mod$n_iter             <- n_done
  mod$elbo_trace         <- elbo_trace[seq_len(n_done)]
  mod$label_loglik_trace <- label_loglik_trace[seq_len(n_done)]

  mod$D  <- D
  mod$V  <- V
  mod$K  <- K
  mod$NZ <- NZ

  mod
}



#' Multi-level supervised topic model (MLSTM) via variational inference.
#'
#' This function fits a multi-output supervised LDA model with a hierarchical
#' prior on regression coefficients:
#'   \deqn{\eta_j \sim N(\mu, \Lambda^{-1}), \quad \Lambda \sim \text{IW}(\upsilon, \Omega).}
#'
#' The latent topic layer is standard LDA, and each response dimension j
#' follows a Gaussian regression on document-level topic proportions.
#' Variational inference is performed by repeated calls to the C++ routine
#' \code{stm_multi_hier_vi_parallel()} until convergence or a maximum
#' number of sweeps is reached.
#'
#' Convergence is assessed based on the relative changes in the evidence lower
#' bound (ELBO) and the supervised label log-likelihood:
#'   \deqn{
#'     \frac{\mathrm{ELBO}_t - \mathrm{ELBO}_{t-1}}{|\mathrm{ELBO}_{t-1}|},
#'     \qquad
#'     \frac{\ell_t - \ell_{t-1}}{|\ell_{t-1}|}.
#'   }
#' After a minimum number of iterations, the algorithm is declared to have
#' converged when both quantities are non-negative and smaller than the
#' prescribed tolerance.
#'
#' @param count Integer matrix with 3 columns (d, v, c), using 0-based indices.
#'   Each row represents document index \code{d}, word index \code{v}, and
#'   token count \code{c}.
#' @param Y Numeric matrix of size D x J containing J response variables
#'   for each of the D documents. NA values are allowed and are ignored
#'   in the initial regression used to seed \code{eta} and \code{sigma2}.
#' @param K Integer, number of topics. Required if \code{phi} is \code{NULL};
#'   ignored if \code{phi} is supplied, in which case \code{K = ncol(phi)}.
#' @param alpha Dirichlet prior parameter for document-topic distributions.
#' @param beta Dirichlet prior parameter for topic-word distributions.
#' @param mu Numeric vector of length K; prior mean for each \eqn{\eta_j}.
#' @param upsilon Scalar degrees of freedom for the inverse-Wishart prior
#'   on the precision matrix \eqn{\Lambda}.
#' @param Omega Numeric K x K positive-definite scale matrix for the
#'   inverse-Wishart prior.
#' @param phi Optional numeric matrix of size V x K used only to initialize
#'   topic assignments via \code{init_mod_from_count()}.
#' @param seed Optional integer random seed used for initialization.
#' @param max_iter Maximum number of variational sweeps.
#' @param min_iter Minimum number of sweeps before checking convergence.
#' @param tol_elbo Numeric tolerance for the relative ELBO change used in
#'   the convergence criterion.
#' @param update_sigma Logical; if TRUE, update \code{sigma2} inside
#'   \code{stm_multi_hier_vi_parallel()}. If FALSE, keep \code{sigma2}
#'   fixed at its initialized value.
#' @param tau Log-space cutoff for local topic responsibilities in the
#'   C++ routine (controls pruning for stability and speed).
#' @param exact_second_moment Logical; reserved flag intended to control whether
#'   the exact second moment \eqn{E[\bar{z}\bar{z}^\top]} is accumulated in the
#'   E-step. **Currently this option has no effect**: the underlying C++
#'   implementation ignores the accumulated second-moment matrix when updating
#'   the variational parameters, and only an approximate moment based on
#'   \eqn{\bar{z}\bar{z}^\top} is effectively used.
#' @param show_progress Logical; forwarded to \code{stm_multi_hier_vi_parallel()}.
#' @param chunk Integer; number of documents per parallel block in the
#'   C++ E-step.
#' @param verbose Logical; if TRUE, print ELBO and its relative change
#'   at each sweep.
#' @param sigma2_init Optional numeric scalar or length-J vector specifying
#'   the initial noise variances. If \code{NULL}, \code{sigma2} is estimated
#'   for each response dimension by least squares regression of
#'   \code{Y[, j]} on initial topic proportions.
#'
#' @return A list \code{mod} containing (at least):
#'   \describe{
#'     \item{nd}{D x K document-topic counts.}
#'     \item{nw}{K x V topic-word counts.}
#'     \item{ndsum}{Integer vector of length D; document token counts.}
#'     \item{nwsum}{Integer vector of length K; topic token counts.}
#'     \item{eta}{K x J matrix of regression coefficients.}
#'     \item{sigma2}{Length-J vector of noise variances.}
#'     \item{Lambda_E}{K x K posterior mean of \eqn{\Lambda} (if returned by C++).}
#'     \item{IW_upsilon_hat}{Posterior degrees of freedom (if returned by C++).}
#'     \item{IW_Omega_hat}{Posterior scale matrix (if returned by C++).}
#'     \item{phi}{V x K topic-word posterior mean
#'       \eqn{p(w \mid z=k)} computed from \code{nw}.}
#'     \item{theta}{D x K document-topic posterior mean
#'       \eqn{p(z=k \mid d)} computed from \code{nd}.}
#'     \item{elbo}{Final ELBO value.}
#'     \item{label_loglik}{Final label log-likelihood term.}
#'     \item{elbo_trace}{Numeric vector of ELBO values over iterations.}
#'     \item{label_loglik_trace}{Numeric vector of label log-likelihoods.}
#'     \item{n_iter}{Number of sweeps actually performed.}
#'     \item{D}{Number of documents.}
#'     \item{V}{Vocabulary size.}
#'     \item{K}{Number of topics.}
#'     \item{J}{Number of response dimensions.}
#'     \item{NZ}{Number of non-zero (d, v, c) entries.}
#'   }
#'
#' @export
run_mlstm_vi <- function(count,
                          Y,
                          K,
                          alpha,
                          beta,
                          mu,
                          upsilon,
                          Omega,
                          phi                = NULL,
                          seed               = NULL,
                          max_iter           = 200L,
                          min_iter           = 50L,
                          tol_elbo           = 1e-4,
                          update_sigma       = TRUE,
                          tau                = 20L,
                          exact_second_moment = FALSE,
                          show_progress      = TRUE,
                          chunk              = 5000L,
                          verbose            = TRUE,
                          sigma2_init        = NULL) {

  # ----- basic checks ------------------------------------------------------
  if (!is.matrix(Y))
    stop("`Y` must be a numeric matrix with dimensions D x J.")
  if (!is.numeric(Y))
    stop("`Y` must be numeric.")

  # initialize LDA state from count
  init <- init_mod_from_count(count, K = K, phi = phi, seed = seed)

  D  <- init$D
  V  <- init$V
  K  <- init$K
  NZ <- init$NZ

  if (nrow(Y) != D)
    stop("nrow(Y) must equal the number of documents D inferred from `count`.")

  J <- ncol(Y)

  # ndsum is fixed across VI sweeps
  ndsum <- init$ndsum

  # ----- initial eta and sigma2 via least squares on normalized counts -----
  X0 <- init$nd / ndsum   # D x K, used only internally

  eta_init    <- matrix(0, nrow = K, ncol = J)
  sigma2_vec  <- rep(1.0, J)

  for (j in seq_len(J)) {
    yj <- Y[, j]
    ok <- which(!is.na(yj))

    if (length(ok) > K) {
      XtX <- crossprod(X0[ok, , drop = FALSE])
      Xty <- crossprod(X0[ok, , drop = FALSE], yj[ok])
      eta_init[, j] <- solve(XtX + diag(1e-6, K), Xty)

      r  <- yj[ok] - X0[ok, , drop = FALSE] %*% eta_init[, j]
      df <- max(1L, length(ok) - K)
      sigma2_vec[j] <- as.numeric(crossprod(r) / df)
    } else {
      eta_init[, j] <- 0
      sigma2_vec[j] <- 1.0
    }
  }

  # override sigma2 if user supplies initial values
  if (!is.null(sigma2_init)) {
    sigma2_init <- as.numeric(sigma2_init)
    if (length(sigma2_init) == 1L) {
      sigma2_vec[] <- sigma2_init
    } else if (length(sigma2_init) == J) {
      sigma2_vec <- sigma2_init
    } else {
      stop("`sigma2_init` must have length 1 or J (ncol(Y)).")
    }
  }

 # ----- initial mod list passed to C++ ------------------------------------
mod <- list(
  nd     = init$nd,
  nw     = init$nw,
  eta    = eta_init,
  sigma2 = sigma2_vec
)

# ensure types for C++
if (!is.integer(count)) {
  count <- apply(count, 2L, as.integer)
  count <- as.matrix(count)
}
Y_mat <- as.matrix(Y)

max_iter <- as.integer(max_iter)
min_iter <- as.integer(min_iter)
if (max_iter <= 0L) stop("`max_iter` must be positive.")
if (min_iter <= 0L) min_iter <- 1L

# NEW: if you don't want a new argument, just set:
tol_label <- tol_elbo

elbo_trace         <- numeric(max_iter)
label_loglik_trace <- numeric(max_iter)

prev_elbo   <- NA_real_
prev_label  <- NA_real_
delta_elbo  <- NA_real_
delta_label <- NA_real_

# ----- VI sweeps with ELBO + label-loglik based convergence --------------
for (iter in seq_len(max_iter)) {

  mod <- stm_multi_hier_vi_parallel(
    mod      = mod,
    docs     = count,
    y        = Y_mat,
    ndsum    = ndsum,
    NZ       = NZ,
    V        = V,
    K        = K,
    J        = J,
    alpha    = alpha,
    beta     = beta,
    mu       = mu,
    upsilon  = upsilon,
    Omega    = Omega,
    update_sigma        = update_sigma,
    tau                 = tau,
    exact_second_moment = exact_second_moment,
    show_progress       = show_progress,
    chunk               = chunk
  )

  elbo_trace[iter]         <- mod$elbo
  label_loglik_trace[iter] <- mod$label_loglik

  # relative changes
  if (!is.na(prev_elbo) && prev_elbo != 0) {
    delta_elbo <- (mod$elbo - prev_elbo) / abs(prev_elbo)
  } else {
    delta_elbo <- NA_real_
  }

  if (!is.na(prev_label) && prev_label != 0) {
    delta_label <- (mod$label_loglik - prev_label) / abs(prev_label)
  } else {
    delta_label <- NA_real_
  }

  if (verbose) {
    msg <- sprintf(
      "[run_mlstm_vi] iter = %d / %d; ELBO = %.5f; dELBO = %s; label = %.5f; dLabel = %s",
      iter, max_iter, mod$elbo,
      if (is.na(delta_elbo)) "NA" else sprintf("%.5e", delta_elbo),
      mod$label_loglik,
      if (is.na(delta_label)) "NA" else sprintf("%.5e", delta_label)
    )
    message(msg)
  }

  # convergence: after min_iter, both relative changes are small and non-negative
  if (iter >= min_iter &&
      !is.na(delta_elbo)  && delta_elbo  >= 0 && delta_elbo  <= tol_elbo &&
      !is.na(delta_label) && delta_label >= 0 && delta_label <= tol_label) {
    if (verbose) {
      message(sprintf(
        "[run_mlstm_vi] converged at iter %d with dELBO = %.5e and dLabel = %.5e",
        iter, delta_elbo, delta_label
      ))
    }
    break
  }

  prev_elbo  <- mod$elbo
  prev_label <- mod$label_loglik
}

n_done <- iter

  # ----- finalize derived quantities --------------------------------------
  mod$ndsum <- ndsum

  nwsum <- as.integer(rowSums(mod$nw))
  mod$nwsum <- nwsum

  # topic-word posterior mean: V x K
  mod$phi <- t((mod$nw + beta) / (nwsum + V * beta))

  # document-topic posterior mean: D x K
  mod$theta <- (mod$nd + alpha) / (mod$ndsum + K * alpha)

  mod$elbo_trace         <- elbo_trace[seq_len(n_done)]
  mod$label_loglik_trace <- label_loglik_trace[seq_len(n_done)]
  mod$n_iter             <- n_done

  mod$D  <- D
  mod$V  <- V
  mod$K  <- K
  mod$J  <- J
  mod$NZ <- NZ

  mod
}

