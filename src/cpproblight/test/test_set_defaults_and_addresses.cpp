#include <cpproblight.h>


xt::xarray<double> forward(xt::xarray<double> observation)
{
  auto prior_mean = 1;
  auto prior_stddev = std::sqrt(5);
  auto likelihood_stddev = std::sqrt(2);

  cpproblight::setDefaultControl(true);
  cpproblight::setDefaultReplace(false);

  auto normal1 = cpproblight::distributions::Normal(prior_mean, prior_stddev);
  auto mu1 = cpproblight::sample(normal1, "normal1");
  mu1      = cpproblight::sample(normal1, "normal1");

  cpproblight::setDefaultControl(true);
  cpproblight::setDefaultReplace(true);

  auto normal2 = cpproblight::distributions::Normal(mu1, prior_stddev);
  auto mu2 = cpproblight::sample(normal2, "normal2");
  mu2      = cpproblight::sample(normal2, "normal2");

  cpproblight::setDefaultControl(false);
  cpproblight::setDefaultReplace(false);

  auto normal3 = cpproblight::distributions::Normal(mu2, prior_stddev);
  auto mu3 = cpproblight::sample(normal3, "normal3");
  mu3      = cpproblight::sample(normal3, "normal3");

  auto likelihood = cpproblight::distributions::Normal(mu3, likelihood_stddev);
  for (auto & o : observation)
  {
    cpproblight::observe(likelihood, o, "likelihood");
  }

  return mu3;
}


int main(int argc, char *argv[])
{
  auto serverAddress = (argc > 1) ? argv[1] : "tcp://*:5555";
  cpproblight::Model model = cpproblight::Model(forward, "Set defaults and addresses test C++");
  model.startServer(serverAddress);
  return 0;
}
