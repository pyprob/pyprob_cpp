#include <cpproblight.h>

// http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf

double marsaglia(double mean, double stddev)
{
  auto uniform = cpproblight::distributions::Uniform(-1, 1);
  auto s = 1.0, x = 0.0, y = 0.0;
  while (s >= 1)
  {
    x = cpproblight::sample(uniform, true, true)(0);
    y = cpproblight::sample(uniform, true, true)(0);
    s = x*x + y*y;
  }
  return mean + stddev * (x * std::sqrt(-2 * std::log(s) / s));
}


xt::xarray<double> forward(xt::xarray<double> observation)
{
  auto prior_mean = 1;
  auto prior_stddev = std::sqrt(5);
  auto likelihood_stddev = std::sqrt(2);

  auto mu = marsaglia(prior_mean, prior_stddev);

  auto likelihood = cpproblight::distributions::Normal(mu, likelihood_stddev);
  for (auto & o : observation)
  {
    cpproblight::observe(likelihood, o);
  }

  return xt::xarray<double> {mu};
}


int main(int argc, char *argv[])
{
  cpproblight::Model model = cpproblight::Model(forward, "Gaussian with unkown mean (Marsaglia) with replacement C++");
  model.startServer();
  return 0;
}
