#include "cpproblight.h"
#include <stdio.h>
#include <iostream>


namespace cpproblight
{
  // https://softwareengineering.stackexchange.com/questions/213631/is-there-any-alternative-to-function-pointers-in-c


  Model::Model(xt::xarray<double> (*modelFunction)(xt::xarray<double>))
  {
    this->modelFunction = modelFunction;
  }

  void Model::startServer(const std::string& serverAddress)
  {
    this->serverAddress = serverAddress;
    printf("Starting server at: %s\n", this->serverAddress.c_str());

    xt::xarray<double> obs = xt::xarray<double> {1.0};

    xt::xarray<double> ret = modelFunction(obs);

    std::cout << ret(0) << std::endl;
  }

  void Model::stopServer()
  {
    printf("Stopping server at: %s\n", this->serverAddress.c_str());
  }
}
