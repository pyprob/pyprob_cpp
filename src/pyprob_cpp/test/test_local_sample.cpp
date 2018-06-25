#include <pyprob_cpp.h>
#include "xtensor/xadapt.hpp"

int main(int argc, char *argv[])
{
  printf("Sampling from Normal({1, 2, 3}, {0.1, 0.01, 0.001})");
  auto normal = pyprob_cpp::distributions::Normal(xt::xarray<double> {1, 2, 3}, xt::xarray<double> {0.1, 0.01, 0.001});
  for (int i = 0; i < 10; i++)
  {
    std::cout << pyprob_cpp::sample(normal) << std::endl;
  }
  pyprob_cpp::observe(normal, xt::xarray<double> {0, 1, 0});

  printf("\nSampling from Uniform(0, 1)");
  auto uniform = pyprob_cpp::distributions::Uniform(0, 1);
  for (int i = 0; i < 10; i++)
  {
    std::cout << pyprob_cpp::sample(uniform) << std::endl;
  }
  pyprob_cpp::observe(uniform, 0);

  printf("\nSampling from Categorical({0.2, 0.7, 0.1})");
  auto categorical = pyprob_cpp::distributions::Categorical(xt::xarray<double> {0.2, 0.7, 0.1});
  for (int i = 0; i < 10; i++)
  {
    std::cout << pyprob_cpp::sample(categorical) << std::endl;
  }
  pyprob_cpp::observe(categorical, 2);

  printf("\nSampling from Poisson({2, 10, 20})");
  auto poisson = pyprob_cpp::distributions::Poisson(xt::xarray<double> {2, 10, 20});
  for (int i = 0; i < 10; i++)
  {
    std::cout << pyprob_cpp::sample(poisson) << std::endl;
  }
  pyprob_cpp::observe(poisson, xt::xarray<double> {0, 1, 2});

  return 0;
}
