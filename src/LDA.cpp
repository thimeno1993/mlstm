#include <Rcpp.h>
using namespace Rcpp;

//' One Gibbs sampling sweep for LDA (collapsed) using document–term list.
//'
//' This function performs a single collapsed Gibbs sampling pass over all
//' non-zero document–term entries. Each (d, v, count) triple is treated as
//' `count` replicated word tokens sharing the same topic assignment.
//'
//' The state is stored in a list `mod` containing:
//'   \describe{
//'     \item{z}{Integer vector of length NZ; topic assignment for each
//'       (d, v, count) triple.}
//'     \item{nd}{D×K integer matrix; document–topic counts.}
//'     \item{nw}{K×V integer matrix; topic–word counts.}
//'     \item{nwsum}{Integer vector of length K; total word count per topic.}
//'   }
//'
//' @param mod List with current sampler state: \code{z}, \code{nd},
//'   \code{nw}, and \code{nwsum} as described above.
//' @param count IntegerMatrix of size NZ×3, where each row is a triple
//'   (d, v, c) with 0-based indices: document index \code{d}, word index
//'   \code{v}, and count \code{c} for that (doc, word) pair.
//' @param ndsum IntegerVector of length D; total token count per document
//'   (i.e., \code{ndsum[d] = sum_k nd(d,k)}). Updated in place.
//' @param NZ Integer, number of non-zero entries (rows in \code{count}
//'   and length of \code{z}).
//' @param V Integer, vocabulary size.
//' @param K Integer, number of topics.
//' @param alpha Scalar Dirichlet prior parameter for document–topic
//'   distributions \eqn{\theta_d} (symmetric).
//' @param beta Scalar Dirichlet prior parameter for topic–word
//'   distributions \eqn{\phi_k} (symmetric).
//'
//' @return A list with updated state:
//'   \describe{
//'     \item{z}{Updated topic assignment vector (length NZ).}
//'     \item{nd}{Updated D×K document–topic counts.}
//'     \item{nw}{Updated K×V topic–word counts.}
//'     \item{nwsum}{Updated total word counts per topic.}
//'   }
//'
// [[Rcpp::export]]
List eLDA_pass_b_fast(List mod, IntegerMatrix count, IntegerVector ndsum,  
                      int NZ, int V, int K, double alpha, double beta) {

  IntegerVector z = mod["z"];
  IntegerMatrix nd = mod["nd"];
  IntegerMatrix nw = mod["nw"];
  IntegerVector nwsum = mod["nwsum"];
  

  double Kalpha = K * alpha;  int dv, d, v, count_dv, topic;
  double Vbeta = V * beta;
  
  NumericVector p(K);
  
  for (dv = 0; dv < NZ; ++dv) {
    d = count(dv, 0);
    v = count(dv, 1);
    count_dv = count(dv, 2);
    
    topic = z[dv];
    
    // Remove current assignment
    nd(d, topic) -= count_dv;
    ndsum[d] -= count_dv;
    nw(topic, v) -= count_dv;
    nwsum[topic] -= count_dv;
    
    // Compute cumulative probabilities for all topics
    double cum_prob = 0.0;
    for (int k = 0; k < K; ++k) {
      double prob = ((nw(k, v) + beta) / (nwsum[k] + Vbeta)) *
        ((nd(d, k) + alpha) / (ndsum[d] + Kalpha));
      cum_prob += prob;
      p[k] = cum_prob;
    }
    
    // Sample new topic from the normalized cumulative distribution
    double u = R::runif(0.0, 1.0) * cum_prob;
    for (topic = 0; topic < K; ++topic) {
      if (p[topic] > u) break;
    }
    
    // Add new assignment
    nd(d, topic) += count_dv;
    ndsum[d] += count_dv;
    nw(topic, v) += count_dv;
    nwsum[topic] += count_dv;
    
    z[dv] = topic;
    
  }
  
  // === Optional: log-likelihood computation (collapsed LDA) ===
  int D = nd.nrow();
  double ll = 0.0;
  
  // Document–topic term
  for (int d = 0; d < D; ++d) {
    double sum_lgamma = 0.0;
    for (int k = 0; k < K; ++k) {
      sum_lgamma += lgamma(nd(d, k) + alpha);
    }
    ll += sum_lgamma - lgamma(ndsum[d] + K * alpha);
  }
  
  // Topic–word term
  for (int k = 0; k < K; ++k) {
    double sum_lgamma = 0.0;
    for (int v = 0; v < V; ++v) {
      sum_lgamma += lgamma(nw(k, v) + beta);
    }
    ll += sum_lgamma - lgamma(nwsum[k] + V * beta);
  }
  
  return List::create(
    _["z"] = z,
    _["nd"] = nd,
    _["nw"] = nw,
    _["nwsum"] = nwsum,
    _["log_likelihood"] = ll
  );
}
