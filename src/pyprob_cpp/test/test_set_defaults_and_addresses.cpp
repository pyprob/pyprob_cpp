#include <pyprob_cpp.h>


xt::xarray<double> forward()
{
  auto prior_mean = 1;
  auto prior_stddev = std::sqrt(5);
  auto likelihood_stddev = std::sqrt(2);

  pyprob_cpp::setDefaultControl(true);
  pyprob_cpp::setDefaultReplace(false);
  auto normal1 = pyprob_cpp::distributions::Normal(prior_mean, prior_stddev);
  xt::xarray<double> mu1;
  for (int i = 0; i < 2; i++)
  {
     mu1 = pyprob_cpp::sample(normal1, "mu1");
  }

  pyprob_cpp::setDefaultControl(true);
  pyprob_cpp::setDefaultReplace(true);
  auto normal2 = pyprob_cpp::distributions::Normal(mu1, prior_stddev);
  xt::xarray<double> mu2;
  for (int i = 0; i < 2; i++)
  {
    mu2 = pyprob_cpp::sample(normal2, "mu2");
  }

  pyprob_cpp::setDefaultControl(false);
  pyprob_cpp::setDefaultReplace(false);
  auto normal3 = pyprob_cpp::distributions::Normal(mu2, prior_stddev);
  xt::xarray<double> mu3;
  for (int i = 0; i < 2; i++)
  {
    mu3 = pyprob_cpp::sample(normal3, "mu3");
  }

  auto likelihood = pyprob_cpp::distributions::Normal(mu3, likelihood_stddev);
  pyprob_cpp::observe(likelihood, "obs");

  return mu3;
}


int main(int argc, char *argv[])
{
  auto serverAddress = (argc > 1) ? argv[1] : "tcp://*:5555";
  pyprob_cpp::Model model = pyprob_cpp::Model(forward, "Set defaults and addresses test C++");
  model.startServer(serverAddress);
  return 0;
}
