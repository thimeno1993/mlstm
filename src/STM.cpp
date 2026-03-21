// [[Rcpp::depends(RcppArmadillo, BH, RcppParallel)]]
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp> // for boost::math::lgamma
#include <unordered_map>
#include <iomanip>

using namespace Rcpp;
using boost::math::digamma;
using boost::math::lgamma;
using RcppParallel::Worker;
using RcppParallel::parallelReduce;

// ---------------- utils ----------------

/**
 * Apply digamma element-wise to an Armadillo row vector and write to a column vector.
 */
inline void digamma_vec(const arma::rowvec& in, arma::vec& out){
  out.set_size(in.n_elem);
  std::transform(in.begin(), in.end(), out.begin(),
                 [](double v){ return digamma(v); });
}

/**
 * Sparse accumulator for topic-word counts nw (K × V), stored column-wise.
 * Instead of holding the full dense K×V matrix per worker, we keep only
 * columns (word indices v) that appear in the current chunk.
 */
struct NWPartial {
  int K{0};
  std::unordered_map<int, std::vector<double>> col_acc; // v -> K-dimensional counts
  double elbo_phi_words{0.0};                           // contribution from word terms in ELBO

  NWPartial() {}
  explicit NWPartial(int K_) : K(K_) {}

  /**
   * Add "scale * x" to column v of nw.
   */
  inline void add_counts(int v, const arma::vec& x, double scale){
    auto &dst = col_acc[v];
    if (dst.empty()) dst.assign(K, 0.0);
    const double* xp = x.memptr();
    for (int k=0; k<K; ++k) dst[k] += scale * xp[k];
  }

  /**
   * Merge another partial accumulator into this one (used in parallelReduce::join).
   */
  inline void merge(const NWPartial& rhs){
    elbo_phi_words += rhs.elbo_phi_words;
    for (const auto& kv : rhs.col_acc){
      const int v = kv.first;
      const std::vector<double>& src = kv.second;
      auto &dst = col_acc[v];
      if (dst.empty()) dst.assign(K, 0.0);
      for (int k=0; k<K; ++k) dst[k] += src[k];
    }
  }

  /**
   * Flush accumulated sparse counts into the dense nw_new matrix.
   */
  inline void flush_to(arma::mat& nw_new) const {
    for (const auto& kv : col_acc){
      int v = kv.first;
      const std::vector<double>& src = kv.second;
      for (int k=0; k<K; ++k) nw_new(k, v) += src[k];
    }
  }
};

// ---------------- Reducer (processes a document range) ----------------
//
//  - No R-managed memory is accessed inside the parallel region.
//  - Performs the local E-step for φ, θ on a subset of documents.
//  - Updates:
//      * gamma(d,:) = α + nd_new(d,:)
//      * nd_new(d,:)
//      * X(d,:) = nd_new(d,:)/N_d
//  - Accumulates:
//      * topic-word counts (nw) via NWPartial
//      * word-related ELBO term elbo_phi_words
//
struct MainLoopReducer : public Worker {
  const int D, K, V;
  const double alpha;
  const std::vector<int>& ndsum;            // copy from R (doc lengths)
  const double* ElogPhi_ptr;                // K×V: expected log φ (own buffer)
  const double* y_ptr;                      // copy from R (length D)
  const double* eta_ptr;                    // regression coefficients (own buffer)
  const double sigma2;
  const std::vector<std::vector<int>>& doc_rows; // precomputed indices per doc
  const int* vptr;                          // word index per nonzero (length NZ)
  const int* cptr;                          // count per nonzero (length NZ)
  const int tau;                            // log-space cutoff for pruning

  // Write-only buffers: all are own-allocated (no R-managed memory)
  double* gammaP; // D×K
  double* nd_newP;// D×K
  double* XP;     // D×K

  // Armadillo views (column-major)
  arma::mat gamma, nd_new, X;
  arma::mat ElogPhi; // read-only

  // Local sparse accumulator for this reducer
  NWPartial local;

  MainLoopReducer(
    int D_, int K_, int V_,
    double alpha_,
    const std::vector<int>& ndsum_,
    const double* ElogPhi_ptr_,
    const double* y_ptr_,
    const double* eta_ptr_,
    double sigma2_,
    const std::vector<std::vector<int>>& doc_rows_,
    const int* vptr_, const int* cptr_,
    int tau_,
    double* gammaP_, double* nd_newP_, double* XP_
  )
    : D(D_), K(K_), V(V_), alpha(alpha_),
      ndsum(ndsum_), ElogPhi_ptr(ElogPhi_ptr_),
      y_ptr(y_ptr_), eta_ptr(eta_ptr_), sigma2(sigma2_),
      doc_rows(doc_rows_), vptr(vptr_), cptr(cptr_), tau(tau_),
      gammaP(gammaP_), nd_newP(nd_newP_), XP(XP_),
      gamma(gammaP_, D, K, /*copy_aux_mem=*/false, /*strict=*/true),
      nd_new(nd_newP_, D, K, false, true),
      X(XP_, D, K, false, true),
      ElogPhi(const_cast<double*>(ElogPhi_ptr_), K, V, false, true),
      local(K_) {}

  // Split constructor for parallelReduce
  MainLoopReducer(const MainLoopReducer& other, RcppParallel::Split)
    : D(other.D), K(other.K), V(other.V), alpha(other.alpha),
      ndsum(other.ndsum), ElogPhi_ptr(other.ElogPhi_ptr),
      y_ptr(other.y_ptr), eta_ptr(other.eta_ptr), sigma2(other.sigma2),
      doc_rows(other.doc_rows), vptr(other.vptr), cptr(other.cptr), tau(other.tau),
      gammaP(other.gammaP), nd_newP(other.nd_newP), XP(other.XP),
      gamma(gammaP, D, K, false, true),
      nd_new(nd_newP, D, K, false, true),
      X(XP, D, K, false, true),
      ElogPhi(const_cast<double*>(ElogPhi_ptr), K, V, false, true),
      local(other.K) {}

  // Main worker: process documents in [begin, end)
  void operator()(std::size_t begin, std::size_t end){
    arma::vec elogtheta(K), label_vec(K), logphi(K), phi(K), nd_row(K);

    for (size_t d = begin; d < end; ++d){
      const auto& rows = doc_rows[d];
      if (rows.empty()){
        // Empty document: set to prior
        for (int k=0;k<K;++k){
          X(d,k)      = 0.0;
          nd_new(d,k) = 0.0;
          gamma(d,k)  = alpha;
        }
        continue;
      }

      const double invN = 1.0 / std::max(1, ndsum[d]);

      // ---- E[log θ_d] = digamma(gamma_d) − digamma(sum_k gamma_dk)
      double sum_g = 0.0;
      for (int k=0;k<K;++k) sum_g += gamma(d,k);
      const double dig_sum = digamma(sum_g);
      arma::rowvec grow = gamma.row(d);
      digamma_vec(grow, elogtheta);
      elogtheta -= dig_sum;

      nd_row.zeros();

      // Token loop within document d
      for (size_t t=0; t<rows.size(); ++t){
        const int i = rows[t];
        const int v = vptr[i];
        const double cc = static_cast<double>(cptr[i]);

        // --- Supervised part: label-dependent term for φ, based on
        //     current nd_row (token i not yet added) to form residual.
        double pred_minus = 0.0;
        const double* ndp_now = nd_row.memptr();   // i is not yet included
        for (int k=0; k<K; ++k) pred_minus += (ndp_now[k] * invN) * eta_ptr[k];
        const double r_minus = y_ptr[d] - pred_minus;
        const double w  = cc * invN;               // token weight
        const double s1i = w / sigma2;             // linear coefficient
        const double s2i = 0.5 * (w * w) / sigma2; // quadratic coefficient (diag approx)
        for (int k=0;k<K;++k){
          const double ek = eta_ptr[k];
          label_vec[k] = (ek * r_minus) * s1i - (ek * ek) * s2i;
        }

        // Local variational distribution over topics: log φ ∝ Elogθ + Elogβ + label term
        logphi = elogtheta + ElogPhi.col(v) + label_vec;

        const double m = logphi.max();
        const double cutoff = m - tau;
        phi = arma::exp(logphi - m);
        for (int k=0;k<K;++k) if (logphi[k] < cutoff) phi[k] = 0.0;

        double s = arma::accu(phi);
        if (s == 0.0){
          // Fallback: all mass on the maximum component
          arma::uword imax = logphi.index_max();
          phi.zeros(); phi[imax] = 1.0; s = 1.0;
        }
        phi /= s;

        // Word-related ELBO contribution: E_q[log p(z, w | θ, β)] - E_q[log q(z)]
        double entropy_phi = 0.0;
        for (int k=0;k<K;++k){
          const double pk = phi[k];
          if (pk > 1e-300) entropy_phi += pk * std::log(pk);
        }
        const double exp_ll = arma::dot(phi, elogtheta + ElogPhi.col(v));
        local.elbo_phi_words += cc * (exp_ll - entropy_phi);

        // Update document-topic counts and sparse topic-word counts
        nd_row += cc * phi;
        local.add_counts(v, phi, cc);
      }

      const double* ndp = nd_row.memptr();
      for (int k=0;k<K;++k){
        X(d,k)      = ndp[k] * invN;   // normalized topic proportion
        nd_new(d,k) = ndp[k];          // unnormalized doc-topic counts
        gamma(d,k)  = alpha + ndp[k];  // posterior Dirichlet parameter
      }
    }
  }

  // Merge accumulators across workers
  void join(const MainLoopReducer& rhs){
    local.merge(rhs.local);
  }
};

// -----------------------------------------------------------
// Main function: parallel variational inference for supervised LDA
// -----------------------------------------------------------
//
// STM (single response):
//   θ_d ~ Dir(α),   β_k ~ Dir(β)
//   z_{d,n} | θ_d ~ Mult(θ_d),   w_{d,n} | z_{d,n} ~ Mult(β_{z_{d,n}})
//   y_d | z̄_d, η, σ² ~ N(z̄_d^T η, σ²)
//
// This routine performs one VI sweep:
//   - E-step (parallel over documents)
//   - M-step for β (topic-word distributions)
//   - M-step for η, σ² (Gaussian response)
//   - ELBO evaluation
//

//' Variational inference for supervised LDA (single continuous response).
//'
//' The model combines unsupervised topic modeling (LDA) with a Gaussian
//' response on document-level topic proportions.
//'
//' \deqn{y_d \sim N(zbar_d^T eta, sigma^2).}
//'
//' This function performs one variational inference sweep with a parallel
//' document-level E-step and simple updates for the regression parameters.
//'
//' @param mod A list containing the current model state:
//'   \describe{
//'     \item{nd}{D x K matrix of document-topic counts.}
//'     \item{nw}{K x V matrix of topic-word counts.}
//'     \item{eta}{Numeric vector of length K; regression coefficients.}
//'     \item{sigma2}{Scalar noise variance for the Gaussian response.}
//'   }
//' @param docs IntegerMatrix of size NZ x 3, where each row is a triple
//'   (d, v, c) in 0-based indexing: document index d, word index v,
//'   and count c = n_dv. Rows with d outside [0, D-1] are ignored.
//' @param y NumericVector of length D; response y_d for each document.
//' @param ndsum IntegerVector of length D; total token count per document
//'   (that is, ndsum[d] = sum_v n_dv).
//' @param NZ Integer, number of non-zero entries in docs (rows of docs).
//' @param V Integer, vocabulary size.
//' @param K Integer, number of topics.
//' @param alpha Scalar Dirichlet prior parameter for document-topic
//'   distributions theta_d (symmetric prior with parameter alpha).
//' @param beta Scalar Dirichlet prior parameter for topic-word
//'   distributions phi_k (symmetric prior with parameter beta).
//' @param update_sigma Logical; if TRUE, update the noise variance sigma2
//'   from residuals y_d - zbar_d^T eta, otherwise keep sigma2 fixed.
//' @param tau Numeric, log-space cutoff used to prune very small topic
//'   responsibilities phi[d,i,k] for numerical stability and efficiency.
//' @param show_progress Logical; if TRUE, print simple progress output
//'   during the E-step over documents.
//' @param chunk Integer, number of documents to process per parallel
//'   block in the E-step. Larger values reduce overhead but may use
//'   more memory.
//'
//' @return A list with updated variational parameters and diagnostics:
//'   \describe{
//'     \item{nd}{Updated D x K document-topic counts.}
//'     \item{nw}{Updated K x V topic-word counts.}
//'     \item{eta}{Updated K-dimensional regression coefficient vector.}
//'     \item{sigma2}{Updated scalar noise variance.}
//'     \item{elbo}{Scalar evidence lower bound (approximate).}
//'     \item{label_loglik}{Gaussian response log-likelihood component.}
//'   }
//'
// [[Rcpp::export]]
List stm_vi_parallel(List mod, IntegerMatrix docs, NumericVector y, IntegerVector ndsum,
                      int NZ, int V, int K, double alpha, double beta,
                      bool update_sigma = true,
                      int tau = 20,
                      bool show_progress = true, int chunk = 5000){
  // ---- Retrieve data from R (single-thread region) ----
  NumericMatrix ndR  = as<NumericMatrix>(mod["nd"]);
  NumericMatrix nwR  = as<NumericMatrix>(mod["nw"]);
  NumericVector etaR = as<NumericVector>(mod["eta"]);
  double sigma2      = as<double>(mod["sigma2"]);
  const int D = ndR.nrow();
  const double Kalpha = K * alpha, Vbeta = V * beta;

  // ---- Copy all inputs needed in the parallel region to self-managed memory ----
  std::vector<int> ndsum_vec = as< std::vector<int> >(ndsum);
  std::vector<double> y_vec  = as< std::vector<double> >(y);
  arma::vec eta(etaR.begin(), K, /*copy_aux_mem=*/true); // copy of eta

  // Decompose docs (NZ×3) into per-document index and separate v, c arrays
  std::vector<int> v_col(NZ), c_col(NZ);
  std::vector<std::vector<int>> doc_rows(D);
  for (int i=0;i<NZ;++i){
    int d = docs(i,0);
    int v = docs(i,1);
    int c = docs(i,2);
    if (d>=0 && d<D){
      doc_rows[d].push_back(i);
      v_col[i] = v;
      c_col[i] = c;
    }
  }

  // Build E[log φ] from old nwR (own buffer)
  arma::mat ElogPhi(K,V); // local buffer
  for (int k=0;k<K;++k){
    double sum_l = V*beta;
    for (int v=0; v<V; ++v) sum_l += nwR(k,v);
    const double dig_sum_l = digamma(sum_l);
    for (int v=0; v<V; ++v){
      const double lv = beta + nwR(k,v);
      ElogPhi(k,v) = digamma(lv) - dig_sum_l;
    }
  }

  // Parallel work buffers (all self-managed)
  arma::mat gamma(D,K, arma::fill::none);
  arma::mat nd_new(D,K, arma::fill::zeros);
  arma::mat X(D,K,      arma::fill::zeros);
  arma::mat nw_new(K,V, arma::fill::zeros);

  // Initialize gamma using current nd (single-thread region)
  for (int d=0; d<D; ++d)
    for (int k=0;k<K;++k) gamma(d,k) = alpha + ndR(d,k);

  if (chunk <= 0) chunk = 5000;
  if (chunk > D)  chunk = D;
  double elbo_phi_words_total = 0.0;

  int done = 0;
  for (int start = 0; start < D; start += chunk){
    int stop = std::min(D, start + chunk);

    MainLoopReducer reducer(
        D, K, V, alpha,
        ndsum_vec,
        ElogPhi.memptr(),
        y_vec.data(),
        eta.memptr(),
        sigma2,
        doc_rows,
        v_col.data(), c_col.data(),
        tau,
        gamma.memptr(), nd_new.memptr(), X.memptr()
    );

    parallelReduce(start, stop, reducer);

    reducer.local.flush_to(nw_new);
    elbo_phi_words_total += reducer.local.elbo_phi_words;

    done = stop;
    if (show_progress){
      double pct = 100.0 * static_cast<double>(done) / static_cast<double>(D);
      Rcpp::Rcout << "\r[E-step] " << done << " / " << D
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)" << std::flush;
    }
    Rcpp::checkUserInterrupt(); // check user interrupt only in single-thread region
  }
  if (show_progress) Rcpp::Rcout << "\n";


  // ---- After all E-step chunks are finished ----
  // 1) Recompute E[log φ] based on nw_new
  arma::mat ElogPhi_new(K, V);
  for (int k=0; k<K; ++k) {
    double sum_l = V*beta;
    for (int v=0; v<V; ++v) sum_l += nw_new(k,v);
    const double dig_sum_l = digamma(sum_l);
    for (int v=0; v<V; ++v) {
      const double lv = beta + nw_new(k,v);
      ElogPhi_new(k,v) = digamma(lv) - dig_sum_l;
    }
  }

  // 2) Word term correction using difference between new and old ElogPhi:
  //    Δ = Σ_{k,v} nw_new(k,v) * (ElogPhi_new(k,v) − ElogPhi_old(k,v))
  double delta_words = 0.0;
  for (int k=0; k<K; ++k){
    for (int v=0; v<V; ++v){
      delta_words += nw_new(k,v) * (ElogPhi_new(k,v) - ElogPhi(k,v)); // ElogPhi is old
    }
  }

  // Adjust the accumulated elbo_phi_words_total to the new ElogPhi baseline
  elbo_phi_words_total += delta_words;

  // ---- Update η (regression parameter) via ridge-like linear solver ----
  arma::vec yA(y_vec.data(), D, /*copy_aux_mem=*/false, /*strict=*/true);
  arma::mat XtX = X.t() * X;
  arma::vec Xty = X.t() * yA;
  XtX.diag() += 1e-6; // small ridge term for stability

  arma::vec eta_new;
  arma::mat L;
  bool ok = arma::chol(L, XtX, "lower"); // true: success, false: fallback
  if (ok) {
    arma::vec z = arma::solve(arma::trimatl(L), Xty);
    eta_new     = arma::solve(arma::trimatu(L.t()), z);
  } else {
    eta_new     = arma::solve(XtX, Xty, arma::solve_opts::likely_sympd);
  }
  eta = eta_new;

  // ---- Update σ² ----
  if (update_sigma) {
    arma::vec resid = yA - X * eta;
    if (D > K) sigma2 = arma::dot(resid,resid) / static_cast<double>(D - K);
  }

  // ---- ELBO computation ----
  double elbo_theta_dir = 0.0;
  {
    arma::vec elog(K);
    for (int d=0; d<D; ++d){
      double sum_g = 0.0; for (int k=0;k<K;++k) sum_g += gamma(d,k);
      const double dig_sum_g = digamma(sum_g);

      double p_term = lgamma(Kalpha) - K*lgamma(alpha); // log p(θ_d | α)
      double q_term = lgamma(sum_g);                    // log Z(q(θ_d))

      arma::rowvec grow = gamma.row(d);
      digamma_vec(grow, elog);
      elog -= dig_sum_g;

      p_term += (alpha - 1.0) * arma::accu(elog);
      for (int k=0;k<K;++k) q_term -= lgamma(gamma(d,k));
      q_term += arma::accu( (grow.t() - 1.0) % elog );

      elbo_theta_dir += (p_term - q_term);
    }
  }

  double elbo_phi_dir = 0.0;
  for (int k=0; k<K; ++k) {
    double sum_l = Vbeta;
    for (int v=0; v<V; ++v) sum_l += nw_new(k,v);
    const double dig_sum_l = digamma(sum_l);
    double p_term = lgamma(Vbeta) - V * lgamma(beta); // log p(β_k | β)
    double q_term = lgamma(sum_l);                    // log Z(q(β_k))

    for (int v=0; v<V; ++v) {
      double lv = beta + nw_new(k,v);
      const double elog = digamma(lv) - dig_sum_l;
      p_term += (beta - 1.0) * elog;
      q_term -= lgamma(lv);
      q_term += (lv - 1.0) * elog;
    }
    elbo_phi_dir += (p_term - q_term);
  }

  const double log_norm_const = -0.5 * std::log(2.0 * arma::datum::pi * sigma2);
  double label_loglik = 0.0;
  for (int d=0; d<D; ++d){
    double pred = arma::as_scalar( X.row(d) * eta );
    double r = y_vec[d] - pred;
    label_loglik += (log_norm_const - 0.5 * (r*r) / sigma2);
  }

  const double elbo = elbo_phi_words_total + elbo_theta_dir + elbo_phi_dir + label_loglik;

  // ---- Return to R: create R objects and copy data back ----
  NumericMatrix nd_newR(D,K), nw_newR(K,V);
  NumericVector eta_out(K);
  std::copy(nd_new.memptr(), nd_new.memptr() + (size_t)D*K, nd_newR.begin());
  std::copy(nw_new.memptr(), nw_new.memptr() + (size_t)K*V, nw_newR.begin());
  std::copy(eta.begin(), eta.end(), eta_out.begin());

  return List::create(
    _["nd"]     = nd_newR,
    _["nw"]     = nw_newR,
    _["eta"]    = eta_out,
    _["sigma2"] = sigma2,
    _["elbo"]   = elbo,
    _["label_loglik"] = label_loglik
  );
}
