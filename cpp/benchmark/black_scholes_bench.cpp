// Single-thread Black-Scholes call pricing benchmark (same formula as Python).
#include <chrono>
#include <cmath>
#include <cstdio>

static double norm_cdf(double x) {
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

static double bs_call(double S, double K, double T, double r, double q,
                      double sigma) {
  double sigma_sqrt_t = sigma * std::sqrt(T);
  double d1 =
      (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrt_t;
  double d2 = d1 - sigma_sqrt_t;
  return S * std::exp(-q * T) * norm_cdf(d1) -
         K * std::exp(-r * T) * norm_cdf(d2);
}

int main() {
  const long N = 1000000;
  const double S = 100.0, K = 100.0, T = 0.5, r = 0.04, q = 0.01,
               sigma = 0.25;
  auto start = std::chrono::steady_clock::now();
  double sum = 0.0;
  for (long i = 0; i < N; ++i) {
    sum += bs_call(S, K, T, r, q, sigma);
  }
  auto end = std::chrono::steady_clock::now();
  double seconds =
      std::chrono::duration<double>(end - start).count();
  std::printf("paths=%ld seconds=%.6f price=%.10f avg=%.10f\n", N, seconds,
              sum / N, bs_call(S, K, T, r, q, sigma));
  return 0;
}
