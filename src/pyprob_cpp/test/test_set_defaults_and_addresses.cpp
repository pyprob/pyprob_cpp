#include <pyprob_cpp.h>


xt::xarray<double> forward(xt::xarray<double> observation)
{
  auto prior_mean = 1;
  auto prior_stddev = std::sqrt(5);
  auto likelihood_stddev = std::sqrt(2);

  pyprob_cpp::setDefaultControl(true);
  pyprob_cpp::setDefaultReplace(false);

  auto normal1 = pyprob_cpp::distributions::Normal(prior_mean, prior_stddev);
  auto mu1 = pyprob_cpp::sample(normal1, "normal1");
  mu1      = pyprob_cpp::sample(normal1, "normal1");

  pyprob_cpp::setDefaultControl(true);
  pyprob_cpp::setDefaultReplace(true);

  auto normal2 = pyprob_cpp::distributions::Normal(mu1, prior_stddev);
  auto mu2 = pyprob_cpp::sample(normal2, "normal2");
  mu2      = pyprob_cpp::sample(normal2, "normal2");

  pyprob_cpp::setDefaultControl(false);
  pyprob_cpp::setDefaultReplace(false);

  auto normal3 = pyprob_cpp::distributions::Normal(mu2, prior_stddev);
  auto mu3 = pyprob_cpp::sample(normal3, "normal3");
  mu3      = pyprob_cpp::sample(normal3, "normal3");

  auto likelihood = pyprob_cpp::distributions::Normal(mu3, likelihood_stddev);
  for (auto & o : observation)
  {
    pyprob_cpp::observe(likelihood, o, "likelihood");
  }

  return mu3;
}


int main(int argc, char *argv[])
{
  auto serverAddress = (argc > 1) ? argv[1] : "tcp://*:5555";
  pyprob_cpp::Model model = pyprob_cpp::Model(forward, xt::xarray<double> {}, "Set defaults and addresses test C++");
  model.startServer(serverAddress);
  return 0;
}
