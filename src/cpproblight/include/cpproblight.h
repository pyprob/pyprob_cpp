#ifndef CPPROBLIGHT_H
#define CPPROBLIGHT_H
#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"


namespace cpproblight
{
  class Model
  {
    std::string serverAddress;
    xt::xarray<double> (*modelFunction)(xt::xarray<double>);

  public:
    Model(xt::xarray<double> (*modelFunction)(xt::xarray<double>));

    void startServer(const std::string& serverAddress);
    void stopServer();
  };
}

#endif
