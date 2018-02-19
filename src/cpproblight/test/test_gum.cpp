// #include <stdio.h>
// #include <stdlib.h>
#include <cpproblight.h>


xt::xarray<double> forward(xt::xarray<double> observation)
{
  auto uniform = cpproblight::distributions::Uniform(0, 1);
  auto a = cpproblight::sample(uniform);
  return a * observation;
}

int main(int argc, char *argv[])
{
  cpproblight::Model model = cpproblight::Model(forward);
  model.startServer();
  return 0;
}
