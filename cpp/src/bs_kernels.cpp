// Batch Black-Scholes price/Greeks and implied-volatility kernels with a C ABI.
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#include <cmath>
#include <cstddef>
#include <random>

static const double PI = 3.14159265358979323846;

static double norm_pdf(double x) {
  return std::exp(-0.5 * x * x) / std::sqrt(2.0 * PI);
}

static double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

static void bs_greeks(double S, double K, double T, double r, double q,
                      double sigma, int call, double *price, double *delta,
                      double *gamma, double *vega, double *theta, double *rho) {
  const double sq = std::sqrt(T);
  const double vol_t = sigma * sq;
  const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_t;
  const double d2 = d1 - vol_t;
  const double disc_s = std::exp(-q * T);
  const double disc_k = std::exp(-r * T);
  const double nd1 = norm_cdf(d1);
  const double nd2 = norm_cdf(d2);
  const double pdf1 = norm_pdf(d1);
  if (call) {
    *price = S * disc_s * nd1 - K * disc_k * nd2;
    *delta = disc_s * nd1;
    *gamma = disc_s * pdf1 / (S * vol_t);
    *vega = S * disc_s * pdf1 * sq;
    *theta = -(S * disc_s * pdf1 * sigma) / (2.0 * sq) -
             r * K * disc_k * nd2 + q * S * disc_s * nd1;
    *rho = K * T * disc_k * nd2;
  } else {
    *price = K * disc_k * norm_cdf(-d2) - S * disc_s * norm_cdf(-d1);
    *delta = -disc_s * norm_cdf(-d1);
    *gamma = disc_s * pdf1 / (S * vol_t);
    *vega = S * disc_s * pdf1 * sq;
    *theta = -(S * disc_s * pdf1 * sigma) / (2.0 * sq) +
             r * K * disc_k * norm_cdf(-d2) - q * S * disc_s * norm_cdf(-d1);
    *rho = -K * T * disc_k * norm_cdf(-d2);
  }
}

static double bs_price_only(double S, double K, double T, double r, double q,
                            double sigma, int call) {
  double price, delta, gamma, vega, theta, rho;
  bs_greeks(S, K, T, r, q, sigma, call, &price, &delta, &gamma, &vega, &theta,
            &rho);
  return price;
}

extern "C" {

EXPORT void batch_bs(
    const double *spot, const double *strike, const double *t,
    const double *r, const double *q, const double *vol, const int *call,
    std::size_t n, double *out_price, double *out_delta, double *out_gamma,
    double *out_vega, double *out_theta, double *out_rho) {
  for (std::size_t i = 0; i < n; ++i) {
    bs_greeks(spot[i], strike[i], t[i], r[i], q[i], vol[i], call[i],
              &out_price[i], &out_delta[i], &out_gamma[i], &out_vega[i],
              &out_theta[i], &out_rho[i]);
  }
}

EXPORT int batch_iv(
    const double *price, const double *spot, const double *strike,
    const double *t, const double *r, const double *q, const int *call,
    std::size_t n, double tol, int max_iter, double *out_iv) {
  int solved = 0;
  for (std::size_t i = 0; i < n; ++i) {
    double lo = 1e-6;
    double hi = 5.0;
    const double target = price[i];
    double lo_price =
        bs_price_only(spot[i], strike[i], t[i], r[i], q[i], lo, call[i]);
    double hi_price =
        bs_price_only(spot[i], strike[i], t[i], r[i], q[i], hi, call[i]);
    if (target < lo_price || target > hi_price) {
      out_iv[i] = std::nan("");
      continue;
    }
    double vol = 0.2;
    for (int it = 0; it < 3 && std::isfinite(vol); ++it) {
      double price_at_vol =
          bs_price_only(spot[i], strike[i], t[i], r[i], q[i], vol, call[i]);
      double delta, gamma, vega, theta, rho;
      bs_greeks(spot[i], strike[i], t[i], r[i], q[i], vol, call[i], &price_at_vol,
                &delta, &gamma, &vega, &theta, &rho);
      double next = vol - (price_at_vol - target) / (vega + 1e-12);
      if (!(next > 0.0) || !(next < 5.0)) break;
      vol = next;
    }
    // Bisection refinement to requested tolerance.
    double mid_price;
    for (int it = 0; it < max_iter; ++it) {
      const double mid = 0.5 * (lo + hi);
      mid_price = bs_price_only(spot[i], strike[i], t[i], r[i], q[i], mid,
                                call[i]);
      if (std::fabs(mid_price - target) < tol) {
        vol = mid;
        break;
      }
      if (mid_price < target)
        lo = mid;
      else
        hi = mid;
      vol = mid;
    }
    out_iv[i] = vol;
    ++solved;
  }
  return solved;
}

EXPORT void mc_gbm(
    double spot, double strike, double T, double r, double q, double sigma,
    int n_paths, unsigned seed, int call, double *out_price, double *out_se) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  const double drift =
      (r - q - 0.5 * sigma * sigma) * T;
  const double diffusion = sigma * std::sqrt(T);
  const double discount = std::exp(-r * T);
  double sum = 0.0;
  double sum_sq = 0.0;
  for (int i = 0; i < n_paths; ++i) {
    const double terminal =
        spot * std::exp(drift + diffusion * normal(rng));
    double payoff = 0.0;
    if (call)
      payoff = terminal > strike ? terminal - strike : 0.0;
    else
      payoff = strike > terminal ? strike - terminal : 0.0;
    const double pnl = discount * payoff;
    sum += pnl;
    sum_sq += pnl * pnl;
  }
  const double mean = sum / n_paths;
  const double variance =
      (sum_sq - n_paths * mean * mean) / (n_paths - 1.0);
  *out_price = mean;
  *out_se = std::sqrt(variance / n_paths);
}

}  // extern "C"
