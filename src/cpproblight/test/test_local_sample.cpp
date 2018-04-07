#include <cpproblight.h>
#include "xtensor/xadapt.hpp"

int main(int argc, char *argv[])
{
  std::cout << "Sampling from Normal({1, 2, 3}, {0.1, 0.01, 0.001})" << std::endl;
  auto normal = cpproblight::distributions::Normal(xt::adapt(std::vector<double>{1, 2, 3}), xt::adapt(std::vector<double>{0.1, 0.01, 0.001}));
  for (int i = 0; i < 10; i++)
  {
    std::cout << cpproblight::sample(normal) << std::endl;
  }
  cpproblight::observe(normal, 0);

  std::cout << "Sampling from Uniform(0, 1)" << std::endl;
  auto uniform = cpproblight::distributions::Uniform(0, 1);
  for (int i = 0; i < 10; i++)
  {
    std::cout << cpproblight::sample(uniform) << std::endl;
  }
  cpproblight::observe(uniform, 0);

  std::cout << "Sampling from Categorical({0.2, 0.7, 0.1})" << std::endl;
  auto categorical = cpproblight::distributions::Categorical(xt::xarray<double> {0.2, 0.7, 0.1});
  for (int i = 0; i < 10; i++)
  {
    std::cout << cpproblight::sample(categorical) << std::endl;
  }
  cpproblight::observe(categorical, 2);

  return 0;
}
