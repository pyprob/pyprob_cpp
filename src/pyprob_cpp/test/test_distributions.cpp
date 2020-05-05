#include <pyprob_cpp.h>

// Model for testing all available distributions

xt::xarray<double> forward()
{
  // Normal
  auto normal_mean = 1.75;
  auto normal_stddev = 0.5;
  auto normal = pyprob_cpp::distributions::Normal(normal_mean, normal_stddev);
  auto normal_sample = pyprob_cpp::sample(normal, "normal");

  // Uniform
  auto uniform_low = 1.2;
  auto uniform_high = 2.5;
  auto uniform = pyprob_cpp::distributions::Uniform(uniform_low, uniform_high);
  auto uniform_sample = pyprob_cpp::sample(uniform, "uniform");

  // Categorical
  auto categorical_probs = xt::xarray<double> {0.1, 0.5, 0.4};
  auto categorical = pyprob_cpp::distributions::Categorical(categorical_probs);
  auto categorical_sample = pyprob_cpp::sample(categorical, "categorical");

  // Poisson
  auto poisson_rate = 4.0;
  auto poisson = pyprob_cpp::distributions::Poisson(poisson_rate);
  auto poisson_sample = pyprob_cpp::sample(poisson, "poisson");

  // Bernoulli
  auto bernoulli_probs = 0.2;
  auto bernoulli = pyprob_cpp::distributions::Bernoulli(bernoulli_probs);
  auto bernoulli_sample = pyprob_cpp::sample(bernoulli, "bernoulli");

  // Beta
  auto beta_concentration1 = 1.2;
  auto beta_concentration0 = 2.5;
  auto beta = pyprob_cpp::distributions::Beta(beta_concentration1, beta_concentration0);
  auto beta_sample = pyprob_cpp::sample(beta, "beta");

  // Exponential
  auto exponential_rate = 2.2;
  auto exponential = pyprob_cpp::distributions::Exponential(exponential_rate);
  auto exponential_sample = pyprob_cpp::sample(exponential, "exponential");

  // Gamma
  auto gamma_concentration = 0.5;
  auto gamma_rate = 1.2;
  auto gamma = pyprob_cpp::distributions::Gamma(gamma_concentration, gamma_rate);
  auto gamma_sample = pyprob_cpp::sample(gamma, "gamma");

  // LogNormal
  auto log_normal_loc = 0.5;
  auto log_normal_scale = 0.2;
  auto log_normal = pyprob_cpp::distributions::LogNormal(log_normal_loc, log_normal_scale);
  auto log_normal_sample = pyprob_cpp::sample(log_normal, "log_normal");

  // Binomial
  auto binomial_total_count = 10.;
  auto binomial_probs = 0.72;
  auto binomial = pyprob_cpp::distributions::Binomial(binomial_total_count, binomial_probs);
  auto binomial_sample = pyprob_cpp::sample(binomial, "binomial");

  // Weibull
  auto weibull_scale = 1.1;
  auto weibull_concentration = 0.6;
  auto weibull = pyprob_cpp::distributions::Weibull(weibull_scale, weibull_concentration);
  auto weibull_sample = pyprob_cpp::sample(weibull, "weibull");

  return 0;
}


int main(int argc, char *argv[])
{
  auto serverAddress = (argc > 1) ? argv[1] : "tcp://*:5555";
  pyprob_cpp::Model model = pyprob_cpp::Model(forward, "Distributions test model C++");
  model.startServer(serverAddress);
  return 0;
}
