// #include <stdio.h>
// #include <stdlib.h>
#include <cpproblight.h>


xt::xarray<double> forward(xt::xarray<double>)
{
  return xt::xarray<double> {1.5,2.0,3.0,4.0};
}

int main(int argc, char *argv[])
{
  cpproblight::Model model = cpproblight::Model(forward);
  model.startServer("server_address");
  model.stopServer();
  return 0;
}
