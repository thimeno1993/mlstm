// [[Rcpp::depends(RcppArmadillo, BH, RcppParallel)]]
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <unordered_map>
#include <iomanip>

using namespace Rcpp;
using namespace RcppParallel;
using boost::math::digamma;
using boost::math::lgamma;

// -----------------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------------

/// Compute digamma for each element in a row (arma::rowvec -> arma::vec).
inline void digamma_row(const arma::rowvec& in, arma::vec& out) {
  out.set_size(in.n_elem);
  std::transform(
    in.begin(), in.end(), out.begin(),
    [](double v){ return digamma(v); }
  );
}

/// Sparse accumulator for topic–word counts nw (K × V).
/// Instead of storing a dense K × V matrix during the E-step, we accumulate
/// per-word K-vectors in a hash map and flush them to nw at the end.
struct NWPartial {
  int K{0};
  std::unordered_map<int, std::vector<double>> acc;  // key: word id v
  double elbo_words{0.0};

  NWPartial() {}
  explicit NWPartial(int K_) : K(K_) {}

  /// Add contribution `scale * phi` for word v.
  inline void add(int v, const arma::vec& phi, double scale) {
    auto &dst = acc[v];
    if (dst.empty()) dst.assign(K, 0.0);
    const double* p = phi.memptr();
    for (int k = 0; k < K; ++k) dst[k] += scale * p[k];
  }

  /// Merge another partial accumulator.
  inline void merge(const NWPartial& rhs) {
    elbo_words += rhs.elbo_words;
    for (const auto& kv : rhs.acc) {
      auto &dst = acc[kv.first];
      if (dst.empty()) dst.assign(K, 0.0);
      const auto& src = kv.second;
      for (int k = 0; k < K; ++k) dst[k] += src[k];
    }
  }

  /// Write accumulated counts into a dense nw matrix (in-place addition).
  inline void flush_to(arma::mat& nw) const {
    for (const auto& kv : acc) {
      int v = kv.first;
      const auto& src = kv.second;
      for (int k = 0; k < K; ++k) nw(k, v) += src[k];
    }
  }
};

// -----------------------------------------------------------------------------
// Parallel document worker (E-step)
// -----------------------------------------------------------------------------
//
// For a block of documents, this worker
//  - updates variational Dirichlet parameters gamma (D × K) for θ_d,
//  - accumulates document-topic counts nd_new (D × K),
//  - computes normalized topic proportions X = nd_new / n_d,
//  - optionally accumulates the total second moment Σ_d Σ_i φ_{d,i} φ_{d,i}^T,
//  - accumulates sparse topic–word counts via NWPartial.
//
struct DocWorker : public Worker {
  // Problem dimensions
  const int D, K, V, J;

  // Hyperparameters / data
  const double alpha;
  const std::vector<int>& ndsum;      // total token count per document (length D)
  const double* ElogPhiP;             // K × V, column-major
  const double* yP;                   // D × J, column-major
  const double* etaP;                 // K × J, column-major
  const double* sigma2P;              // length J

  // Sparse document representation
  // - doc_rows[d] : indices (0..NZ-1) into vcol/ccol belonging to doc d
  // - vcol[idx]   : word index v for non-zero entry idx
  // - ccol[idx]   : count of word v in its document for entry idx
  const std::vector<std::vector<int>>& doc_rows;
  const int* vcol;                    // length NZ
  const int* ccol;                    // length NZ

  // Local inference options
  const int tau;                      // log-cutoff for pruning topics
  const bool exact_second_moment;     // if true, accumulate Σ φ φᵀ

  // Shared buffers (views on top of raw pointers)
  double* gammaP;                     // D × K
  double* nd_newP;                    // D × K
  double* XP;                         // D × K

  // Optional: global accumulator for second moments (K × K).
  // Note: sum_outer_total is currently only accumulated in the E-step;
  //       the outer algorithm does not yet use these second moments.
  //       The flag `exact_second_moment` is therefore reserved for
  //       future extensions.
  arma::mat* sum_outer_total;

  // Matrix views (not owning data)
  arma::mat ElogPhi;                  // K × V
  arma::mat gamma;                    // D × K
  arma::mat nd_new;                   // D × K
  arma::mat X;                        // D × K

  // Local, per-chunk accumulator for nw and word-related ELBO
  NWPartial local;

  DocWorker(
    int D_, int K_, int V_, int J_,
    double alpha_, const std::vector<int>& ndsum_,
    const double* ElogPhiP_, const double* yP_,
    const double* etaP_, const double* sigma2P_,
    const std::vector<std::vector<int>>& doc_rows_,
    const int* vcol_, const int* ccol_,
    int tau_, bool exact_second_moment_,
    double* gammaP_, double* nd_newP_, double* XP_,
    arma::mat* sum_outer_total_
  )
    : D(D_), K(K_), V(V_), J(J_),
      alpha(alpha_), ndsum(ndsum_),
      ElogPhiP(ElogPhiP_), yP(yP_), etaP(etaP_), sigma2P(sigma2P_),
      doc_rows(doc_rows_), vcol(vcol_), ccol(ccol_),
      tau(tau_), exact_second_moment(exact_second_moment_),
      gammaP(gammaP_), nd_newP(nd_newP_), XP(XP_),
      sum_outer_total(sum_outer_total_),
      ElogPhi(const_cast<double*>(ElogPhiP_), K, V, false, true),
      gamma(gammaP_, D, K, false, true),
      nd_new(nd_newP_, D, K, false, true),
      X(XP_, D, K, false, true),
      local(K_)
  {}

  // Split constructor for parallelReduce
  DocWorker(const DocWorker& o, Split)
    : D(o.D), K(o.K), V(o.V), J(o.J),
      alpha(o.alpha), ndsum(o.ndsum),
      ElogPhiP(o.ElogPhiP), yP(o.yP), etaP(o.etaP), sigma2P(o.sigma2P),
      doc_rows(o.doc_rows), vcol(o.vcol), ccol(o.ccol),
      tau(o.tau), exact_second_moment(o.exact_second_moment),
      gammaP(o.gammaP), nd_newP(o.nd_newP), XP(o.XP),
      sum_outer_total(o.sum_outer_total),
      ElogPhi(const_cast<double*>(ElogPhiP), K, V, false, true),
      gamma(gammaP, D, K, false, true),
      nd_new(nd_newP, D, K, false, true),
      X(XP, D, K, false, true),
      local(o.K)
  {}

  /// Main worker: process documents [begin, end).
  void operator()(std::size_t begin, std::size_t end) {
    arma::vec elogtheta(K), logphi(K), phi(K), ndrow(K);

    // Local second-moment accumulator, if requested
    arma::mat sum_outer_loc;
    if (exact_second_moment) sum_outer_loc.zeros(K, K);

    for (size_t d = begin; d < end; ++d) {
      const auto& rows = doc_rows[d];

      // Empty document: reset gamma, nd_new, X
      if (rows.empty()) {
        for (int k = 0; k < K; ++k) {
          gamma(d, k)  = alpha;
          nd_new(d, k) = 0.0;
          X(d, k)      = 0.0;
        }
        continue;
      }

      const double invN = 1.0 / std::max(1, ndsum[d]);

      // E_q[log θ_d]
      double sumg = 0.0;
      for (int k = 0; k < K; ++k) sumg += gamma(d, k);
      const double dig_sumg = digamma(sumg);
      arma::rowvec grow = gamma.row(d);
      digamma_row(grow, elogtheta);
      elogtheta -= dig_sumg;

      ndrow.zeros();

      // Loop over nonzero words in document d
      for (size_t t = 0; t < rows.size(); ++t) {
        const int idx = rows[t];
        const int v   = vcol[idx];
        const double c = static_cast<double>(ccol[idx]);

        // Approximate label-dependent contribution for this word:
        // aggregated over all J responses (multi-output regression).
        arma::vec label_term(K, arma::fill::zeros);
        const double w = c * invN;

        for (int j = 0; j < J; ++j) {
          const double s2  = sigma2P[j];
          const double s1i = w / s2;
          const double s2i = 0.5 * (w * w) / s2;

          // pred_minus ≈ (ndrow / n_d)·eta(:, j)
          double pred_minus = 0.0;
          for (int k = 0; k < K; ++k)
            pred_minus += (ndrow[k] * invN) * etaP[k + K * j];

          const double r_minus = yP[d + D * j] - pred_minus;

          for (int k = 0; k < K; ++k) {
            const double ek = etaP[k + K * j];
            label_term[k] += ek * r_minus * s1i - (ek * ek) * s2i;
          }
        }

        // Variational distribution over topics for this token: log φ ∝ log θ + log β + label term.
        logphi = elogtheta + ElogPhi.col(v) + label_term;

        // Numerical stabilization & pruning: keep topics within tau of maximum.
        const double m      = logphi.max();
        const double cutoff = m - tau;
        phi = arma::exp(logphi - m);
        for (int k = 0; k < K; ++k)
          if (logphi[k] < cutoff) phi[k] = 0.0;

        double s = arma::accu(phi);
        if (s == 0.0) {
          arma::uword imax = logphi.index_max();
          phi.zeros();
          phi[imax] = 1.0;
          s = 1.0;
        }
        phi /= s;

        // Word-related contribution to ELBO (entropy and expected log likelihood)
        double entropy_phi = 0.0;
        for (int k = 0; k < K; ++k)
          if (phi[k] > 1e-300) entropy_phi += phi[k] * std::log(phi[k]);

        const double exp_ll = arma::dot(phi, elogtheta + ElogPhi.col(v));
        local.elbo_words += c * (exp_ll - entropy_phi);

        // Update document-topic counts
        ndrow += c * phi;

        // Optional second moment Σ φ φᵀ
        if (exact_second_moment)
          sum_outer_loc += c * (phi * phi.t());

        // Sparse topic–word counts
        local.add(v, phi, c);
      }

      // Update nd_new, normalized X, and Dirichlet gamma
      const double* np = ndrow.memptr();
      for (int k = 0; k < K; ++k) {
        nd_new(d, k) = np[k];
        X(d, k)      = np[k] * invN;      // E[z̄_d]
        gamma(d, k)  = alpha + np[k];     // posterior shape
      }
    }

    // Reduction for second moment (if enabled).
    if (exact_second_moment) {
#pragma omp critical
      {
        (*sum_outer_total) += sum_outer_loc;
      }
    }
  }

  /// Combine NWPartial from another worker (parallelReduce).
  void join(const DocWorker& rhs) {
    local.merge(rhs.local);
  }
};

// -----------------------------------------------------------------------------
// Main routine
// -----------------------------------------------------------------------------

//' Variational inference for multi-output supervised topic models
//' with hierarchical prior.
//'
//' The model includes:
//'   - LDA structure: theta_d ~ Dir(alpha), phi_k ~ Dir(beta)
//'   - Gaussian response: y[d,j] ~ N(zbar_d^T eta_j, sigma_j^2)
//'   - Hierarchical prior:
//'       eta_j ~ N(mu, Lambda^-1)
//'       Lambda ~ inverse-Wishart(upsilon, Omega)
//'
//' @param mod List with model state:
//'   - nd  (D x K) document-topic counts
//'   - nw  (K x V) topic-word counts
//'   - eta (K x J) regression coefficients
//'   - sigma2 (J) noise variances
//' @param docs IntegerMatrix (NZ x 3) with (doc_id, word_id, count).
//' @param y NumericMatrix (D x J) response matrix.
//' @param ndsum IntegerVector (D) document token counts.
//' @param NZ,V,K,J Model dimensions.
//' @param alpha,beta Dirichlet hyperparameters.
//' @param mu NumericVector (K) prior mean.
//' @param upsilon Degrees of freedom for inverse-Wishart.
//' @param Omega Scale matrix for inverse-Wishart.
//' @param update_sigma Logical; update sigma2 or not.
//' @param tau Numeric cutoff for stability.
//' @param exact_second_moment Logical flag (currently not used).
//' @param show_progress Logical; print progress.
//' @param chunk Integer; documents per parallel block.
//'
//' @return A list with updated variational parameters and diagnostics:
//'   \describe{
//'     \item{nd}{D x K integer matrix of document-topic counts.}
//'     \item{nw}{K x V integer matrix of topic-word counts.}
//'     \item{eta}{K x J numeric matrix of regression coefficients.}
//'     \item{sigma2}{Length-J numeric vector of noise variances.}
//'     \item{Lambda_E}{K x K numeric matrix, posterior mean of precision matrix Lambda.}
//'     \item{IW_upsilon_hat}{Numeric scalar, posterior degrees of freedom.}
//'     \item{IW_Omega_hat}{K x K numeric matrix, posterior scale matrix.}
//'     \item{elbo}{Numeric scalar, evidence lower bound.}
//'     \item{label_loglik}{Numeric scalar, supervised log-likelihood term.}
//'   }
// [[Rcpp::export]]
List stm_multi_hier_vi_parallel(
  List mod,
  IntegerMatrix docs,
  NumericMatrix y,
  IntegerVector ndsum,
  int NZ, int V, int K, int J,
  double alpha, double beta,
  NumericVector mu, double upsilon, NumericMatrix Omega,
  bool update_sigma      = true,
  int  tau               = 20,
  bool exact_second_moment = false,
  bool show_progress     = true,
  int  chunk             = 5000
) {
  // -----------------------------
  // Input / basic checks
  // -----------------------------
  NumericMatrix ndR      = as<NumericMatrix>(mod["nd"]);       // D × K
  NumericMatrix nwR      = as<NumericMatrix>(mod["nw"]);       // K × V
  NumericMatrix etaR     = as<NumericMatrix>(mod["eta"]);      // K × J
  NumericVector sigma2R  = as<NumericVector>(mod["sigma2"]);   // J

  const int D = ndR.nrow();

  if (ndR.ncol() != K || nwR.nrow() != K || nwR.ncol() != V)
    stop("nd/nw size mismatch");
  if (etaR.nrow() != K || etaR.ncol() != J)
    stop("eta size mismatch");
  if ((int)sigma2R.size() != J)
    stop("sigma2 size mismatch");
  if (y.nrow() != D || y.ncol() != J)
    stop("y size mismatch");
  if ((int)mu.size() != K)
    stop("mu size mismatch");
  if (Omega.nrow() != K || Omega.ncol() != K)
    stop("Omega size mismatch");

  // Wrap as Armadillo views
  arma::mat ndA(ndR.begin(), D, K, true, true);
  arma::mat nwA(nwR.begin(), K, V, true, true);
  arma::mat eta(etaR.begin(), K, J, true, true);
  arma::vec sigma2(sigma2R.begin(), J, true, true);
  arma::vec muA(mu.begin(), K, true, true);
  arma::mat OmegaA(Omega.begin(), K, K, true, true);
  std::vector<int> ndsum_vec = as<std::vector<int>>(ndsum);
  arma::mat yA(y.begin(), D, J, true, true);

  const double Kalpha = K * alpha;
  const double Vbeta  = V * beta;

  // -----------------------------
  // Build sparse doc structure
  // -----------------------------
  std::vector<int> vcol(NZ), ccol(NZ);
  std::vector<std::vector<int>> doc_rows(D);

  for (int i = 0; i < NZ; ++i) {
    int d = docs(i, 0);
    int v = docs(i, 1);
    int c = docs(i, 2);
    if (d >= 0 && d < D) {
      doc_rows[d].push_back(i);
      vcol[i] = v;
      ccol[i] = c;
    }
  }

  // -----------------------------
  // E[log φ] from current nwA
  // -----------------------------
  arma::mat ElogPhi(K, V);
  for (int k = 0; k < K; ++k) {
    double suml = V * beta;
    for (int v = 0; v < V; ++v) suml += nwA(k, v);
    const double digs = digamma(suml);
    for (int v = 0; v < V; ++v) {
      double lv = beta + nwA(k, v);
      ElogPhi(k, v) = digamma(lv) - digs;
    }
  }

  // -----------------------------
  // Parallel work buffers
  // -----------------------------
  arma::mat gamma(D, K, arma::fill::none);   // Dirichlet parameters for θ_d
  arma::mat nd_new(D, K, arma::fill::zeros); // updated document-topic counts
  arma::mat X(D, K, arma::fill::zeros);      // normalized topic proportions
  arma::mat nw_new(K, V, arma::fill::zeros); // new topic-word counts

  // Accumulated word-related ELBO contribution from all documents
  double elbo_words_total = 0.0;

  for (int d = 0; d < D; ++d)
    for (int k = 0; k < K; ++k)
      gamma(d, k) = alpha + ndA(d, k);

  arma::mat sum_outer_total; // used only if exact_second_moment == true
  if (exact_second_moment) sum_outer_total.zeros(K, K);

  // -----------------------------
  // Parallel E-step over documents
  // -----------------------------
  // Note: when `exact_second_moment = true`, the worker accumulates
  //       φ φᵀ contributions into sum_outer_total, but these second
  //       moments are not yet used in the M-step updates. The flag
  //       is kept for future extensions.
  if (chunk <= 0) chunk = 5000;

  for (int start = 0; start < D; start += chunk) {
    int stop = std::min(D, start + chunk);

    DocWorker w(
      D, K, V, J,
      alpha, ndsum_vec,
      ElogPhi.memptr(), yA.memptr(), eta.memptr(), sigma2.memptr(),
      doc_rows, vcol.data(), ccol.data(),
      tau, exact_second_moment,
      gamma.memptr(), nd_new.memptr(), X.memptr(),
      &sum_outer_total
    );

    parallelReduce(start, stop, w);

    // Accumulate local nw contributions from this chunk
    w.local.flush_to(nw_new);

    // Accumulate word-related ELBO contribution
    elbo_words_total += w.local.elbo_words;

    if (show_progress) {
      double pct = 100.0 * static_cast<double>(stop) / static_cast<double>(D);
      Rcpp::Rcout << "\r[E-step] " << stop << "/" << D
                  << " (" << std::fixed << std::setprecision(1)
                  << pct << "%)" << std::flush;
    }
    Rcpp::checkUserInterrupt();
  }
  if (show_progress) Rcpp::Rcout << "\n";

  // -----------------------------
  // Recompute E[log φ] for updated nw_new
  // -----------------------------
  arma::mat ElogPhi_new(K, V);
  for (int k = 0; k < K; ++k) {
    double suml = V * beta;
    for (int v = 0; v < V; ++v) suml += nw_new(k, v);
    const double digs = digamma(suml);
    for (int v = 0; v < V; ++v) {
      double lv = beta + nw_new(k, v);
      ElogPhi_new(k, v) = digamma(lv) - digs;
    }
  }

  // ---------------------------------------------------------------------------
  // Hierarchical updates for (η_j, Λ) and σ_j^2
  // ---------------------------------------------------------------------------

  // Indices of observed responses per j (NA-removal)
  std::vector<arma::uvec> ok_rows(J);
  for (int j = 0; j < J; ++j) {
    arma::vec yj = yA.col(j);
    ok_rows[j] = arma::find(yj == yj);  // y == y filters out NAs
  }

  // E[Λ] initial value: prior-based default if no previous iteration info is used.
  arma::mat Lambda_E = OmegaA / std::max(upsilon - K - 1.0, 1.0);

  arma::mat eta_new(K, J, arma::fill::zeros);
  arma::mat eta_se (K, J, arma::fill::zeros);   // standard errors per η_j

  // S accumulates E[(η_j - μ)(η_j - μ)^T] over j, used to update Λ.
  arma::mat S(K, K, arma::fill::zeros);

  // Use prior as "previous" IW parameters (simple single-step update).
  double    upsilon_prev = upsilon;
  arma::mat Omega_prev   = OmegaA;

  arma::mat OmegaPrev_inv;
  bool ok_inv = arma::inv_sympd(OmegaPrev_inv, Omega_prev);
  if (!ok_inv)
    OmegaPrev_inv = arma::pinv(Omega_prev);

  // E[Λ^{-1}] under previous IW (upsilon_prev, Omega_prev)
  arma::mat LambdaPrec_E = upsilon_prev * OmegaPrev_inv;

  // Update each η_j given X and y_{·,j}
  for (int j = 0; j < J; ++j) {
    const arma::uvec& ok = ok_rows[j];
    arma::mat Xj = X.rows(ok);                 // D_ok × K
    arma::vec yj_full = arma::vec(yA.col(j));
    arma::vec yj      = yj_full.elem(ok);

    // S1 = Σ_d y_dj X_d,  S2 ≈ Σ_d X_d X_dᵀ
    arma::vec S1 = Xj.t() * yj;
    arma::mat S2 = Xj.t() * Xj;

    const double s2 = std::max(sigma2(j), 1e-12);
    arma::mat A = (S2 / s2) + LambdaPrec_E;
    arma::vec b = (S1 / s2) + LambdaPrec_E * muA;

    // Numerical stabilization
    A.diag() += 1e-8;

    arma::vec mu_hat_j;
    arma::mat L;
    arma::mat Vj;  // approximate posterior covariance for η_j

    if (arma::chol(L, A, "lower")) {
      arma::vec z = arma::solve(arma::trimatl(L), b);
      mu_hat_j    = arma::solve(arma::trimatu(L.t()), z);

      // V_j = A^{-1} = L^{-T} L^{-1}
      arma::mat I = arma::eye(K, K);
      arma::mat Linv = arma::solve(arma::trimatl(L), I);
      Vj = Linv.t() * Linv;
    } else {
      // Fallback: general solver + inverse
      mu_hat_j = arma::solve(A, b, arma::solve_opts::likely_sympd);

      arma::mat Vj_tmp;
      bool ok_invA = arma::inv_sympd(Vj_tmp, A);
      if (ok_invA) {
        Vj = Vj_tmp;
      } else {
        Vj = arma::pinv(A);
      }
    }

    eta_new.col(j) = mu_hat_j;
    eta_se.col(j)  = arma::sqrt(Vj.diag());

    arma::vec diff = mu_hat_j - muA;
    S += Vj + diff * diff.t();
  }

  // Posterior for Λ under inverse-Wishart:
  //   ν̂ = ν + J,  Ω̂ = Ω + Σ_j [ V_j + (μ̂_j − μ)(μ̂_j − μ)^T ].
  double    upsilon_hat = upsilon + J;
  arma::mat Omega_hat   = OmegaA + S;

  // E_q[Λ] = Ω̂ / (ν̂ − K − 1)
  double denom = std::max(upsilon_hat - K - 1.0, 1.0);
  Lambda_E = Omega_hat / denom;

  // Update σ_j^2 if requested
  if (update_sigma) {
    for (int j = 0; j < J; ++j) {
      const arma::uvec& ok = ok_rows[j];
      arma::mat Xj = X.rows(ok);
      arma::vec yj_full = arma::vec(yA.col(j));
      arma::vec yj      = yj_full.elem(ok);
      arma::vec r       = yj - Xj * eta_new.col(j);
      int df = std::max(1, static_cast<int>(ok.n_elem) - K);
      sigma2(j) = arma::dot(r, r) / static_cast<double>(df);
    }
  }

  // Write back updated eta
  eta = eta_new;

  // ---------------------------------------------------------------------------
  // ELBO components (approximate)
  // ---------------------------------------------------------------------------

  // Contribution from Dirichlet q(θ_d)
  double elbo_theta = 0.0;
  {
    arma::vec elog(K);
    for (int d = 0; d < D; ++d) {
      double sumg = 0.0;
      for (int k = 0; k < K; ++k) sumg += gamma(d, k);

      double p = lgamma(Kalpha) - K * lgamma(alpha);
      double q = lgamma(sumg);
      const double digs = digamma(sumg);

      arma::rowvec grow = gamma.row(d);
      digamma_row(grow, elog);
      elog -= digs;

      p += (alpha - 1.0) * arma::accu(elog);
      for (int k = 0; k < K; ++k) q -= lgamma(gamma(d, k));
      q += arma::accu((grow.t() - 1.0) % elog);

      elbo_theta += (p - q);
    }
  }

  // Contribution from Dirichlet q(φ_k)
  double elbo_phi = 0.0;
  for (int k = 0; k < K; ++k) {
    double suml = Vbeta;
    for (int v = 0; v < V; ++v) suml += nw_new(k, v);

    double p = lgamma(Vbeta) - V * lgamma(beta);
    double q = lgamma(suml);
    const double digs = digamma(suml);

    for (int v = 0; v < V; ++v) {
      double lv   = beta + nw_new(k, v);
      double elog = digamma(lv) - digs;
      p += (beta - 1.0) * elog;
      q -= lgamma(lv);
      q += (lv - 1.0) * elog;
    }
    elbo_phi += (p - q);
  }

  // Gaussian supervised likelihood (ignoring NAs)
  double label_loglik = 0.0;
  arma::vec logc = -0.5 * arma::log(2.0 * arma::datum::pi * sigma2);
  for (int j = 0; j < J; ++j) {
    const arma::uvec& ok = ok_rows[j];
    arma::mat Xj = X.rows(ok);
    arma::vec yj_full = arma::vec(yA.col(j));
    arma::vec yj      = yj_full.elem(ok);
    arma::vec r       = yj - Xj * eta.col(j);
    label_loglik += ok.n_elem * logc(j)
                    - 0.5 * arma::dot(r, r) / sigma2(j);
  }

  double elbo = elbo_theta + elbo_phi + label_loglik + elbo_words_total;

  // ---------------------------------------------------------------------------
  // Return to R
  // ---------------------------------------------------------------------------
  NumericMatrix nd_out(D, K), nw_out(K, V),
                eta_out(K, J), eta_se_out(K, J),
                LambdaE_out(K, K), Omega_hat_out(K, K);
  NumericVector sigma2_out(J);

  std::copy(nd_new.memptr(),    nd_new.memptr()    + (size_t)D * K, nd_out.begin());
  std::copy(nw_new.memptr(),    nw_new.memptr()    + (size_t)K * V, nw_out.begin());
  std::copy(eta.memptr(),       eta.memptr()       + (size_t)K * J, eta_out.begin());
  std::copy(eta_se.memptr(),    eta_se.memptr()    + (size_t)K * J, eta_se_out.begin());
  std::copy(Lambda_E.memptr(),  Lambda_E.memptr()  + (size_t)K * K, LambdaE_out.begin());
  std::copy(Omega_hat.memptr(), Omega_hat.memptr() + (size_t)K * K, Omega_hat_out.begin());
  std::copy(sigma2.memptr(),    sigma2.memptr()    + (size_t)J,     sigma2_out.begin());

  return List::create(
    _["nd"]            = nd_out,
    _["nw"]            = nw_out,
    _["eta"]           = eta_out,        // E_q[η_j] = μ̂_j
    _["eta_se"]        = eta_se_out,     // standard errors for η_j
    _["sigma2"]        = sigma2_out,
    _["Lambda_E"]      = LambdaE_out,    // E_q[Λ]
    _["IW_upsilon_hat"]= upsilon_hat,
    _["IW_Omega_hat"]  = Omega_hat_out,  // posterior IW scale
    _["elbo"]          = elbo,
    _["label_loglik"]  = label_loglik
  );
}
