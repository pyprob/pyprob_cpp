#include <cpproblight.h>

// Branching
// http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf

int fibonacci(int n)
{
    int a = 1, b = 1;
    for (int i = 3; i <= n; i++) {
        int c = a + b;
        a = b;
        b = c;
    }
    return b;
}

xt::xarray<double> forward(xt::xarray<double> observation)
{
  auto count_prior = cpproblight::distributions::Poisson(4);
  auto r = cpproblight::sample(count_prior)(0);

  int l;
  if (4 < r)
  {
    l = 6;
  }
  else
  {
    l = 1 + fibonacci(3 * r) + cpproblight::sample(count_prior)(0);
  }
  auto likelihood = cpproblight::distributions::Poisson(l);
  cpproblight::observe(likelihood, observation);
  return r;
}


int main(int argc, char *argv[])
{
  auto serverAddress = (argc > 1) ? argv[1] : "tcp://*:5555";
  cpproblight::Model model = cpproblight::Model(forward, xt::xarray<double> {}, "Branching C++");
  model.startServer(serverAddress);
  return 0;
}
