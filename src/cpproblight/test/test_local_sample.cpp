#include <cpproblight.h>
#include "xtensor/xadapt.hpp"

int main(int argc, char *argv[])
{
  printf("Sampling from Normal({1, 2, 3}, {0.1, 0.01, 0.001})");
  auto normal = cpproblight::distributions::Normal(xt::xarray<double> {1, 2, 3}, xt::xarray<double> {0.1, 0.01, 0.001});
  for (int i = 0; i < 10; i++)
  {
    std::cout << cpproblight::sample(normal) << std::endl;
  }
  cpproblight::observe(normal, xt::xarray<double> {0, 1, 0});

  printf("\nSampling from Uniform(0, 1)");
  auto uniform = cpproblight::distributions::Uniform(0, 1);
  for (int i = 0; i < 10; i++)
  {
    std::cout << cpproblight::sample(uniform) << std::endl;
  }
  cpproblight::observe(uniform, 0);

  printf("\nSampling from Categorical({0.2, 0.7, 0.1})");
  auto categorical = cpproblight::distributions::Categorical(xt::xarray<double> {0.2, 0.7, 0.1});
  for (int i = 0; i < 10; i++)
  {
    std::cout << cpproblight::sample(categorical) << std::endl;
  }
  cpproblight::observe(categorical, 2);

  printf("\nSampling from Poisson({2, 10, 20})");
  auto poisson = cpproblight::distributions::Poisson(xt::xarray<double> {2, 10, 20});
  for (int i = 0; i < 10; i++)
  {
    std::cout << cpproblight::sample(poisson) << std::endl;
  }
  cpproblight::observe(poisson, xt::xarray<double> {0, 1, 2});

  return 0;
}
