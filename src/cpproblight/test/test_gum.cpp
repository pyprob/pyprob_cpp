// #include <stdio.h>
// #include <stdlib.h>
#include <cpproblight.h>


xt::xarray<double> forward(xt::xarray<double> observation)
{
  auto prior_mean = 1;
  auto prior_stddev = std::sqrt(5);
  auto likelihood_stddev = std::sqrt(2);

  auto prior = cpproblight::distributions::Normal(prior_mean, prior_stddev);
  auto mu = cpproblight::sample(prior);

  auto likelihood = cpproblight::distributions::Normal(mu(0), likelihood_stddev);
  for (auto & o : observation)
  {
    cpproblight::observe(likelihood, o);
  }

  return mu;
}


int main(int argc, char *argv[])
{
  cpproblight::Model model = cpproblight::Model(forward);
  model.startServer();
  return 0;
}
