#include <cpproblight.h>


int main(int argc, char *argv[])
{
  auto normal = cpproblight::distributions::Normal(1, std::sqrt(5));
  auto normal_sample = cpproblight::sample(normal);
  std::cout << normal_sample << std::endl;
  cpproblight::observe(normal, 0);

  auto uniform = cpproblight::distributions::Uniform(0, 1);
  auto uniform_sample = cpproblight::sample(uniform);
  std::cout << uniform_sample << std::endl;
  cpproblight::observe(uniform, 0);

  return 0;
}
