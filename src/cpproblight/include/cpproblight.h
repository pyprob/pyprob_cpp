#ifndef CPPROBLIGHT_H
#define CPPROBLIGHT_H
#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "pplprotocol.h"
#include <zmq.hpp>

#define VERSION "0.1.0"
#define GIT_BRANCH "master"
#define GIT_COMMIT_HASH "8b946d7"

namespace cpproblight
{
  namespace distributions
  {
    class Distribution
    {
    public:
      virtual xt::xarray<double> sample(const bool control, const bool replace, const std::string& address);
      virtual void observe(xt::xarray<double> value, const std::string& address);
    };

    class Uniform: public Distribution
    {
    private:
      double low;
      double high;

    public:
      Uniform(double low=0, double high=1);
      xt::xarray<double> sample(const bool control, const bool replace, const std::string& address);
      void observe(xt::xarray<double> value, const std::string& address);
    };

    class Normal: public Distribution
    {
    private:
      xt::xarray<double> mean;
      xt::xarray<double> stddev;

    public:
      Normal(xt::xarray<double> mean=xt::xarray<double> {0}, xt::xarray<double> stddev=xt::xarray<double> {1});
      xt::xarray<double> sample(const bool control, const bool replace, const std::string& address);
      void observe(xt::xarray<double> value, const std::string& address);
    };
  }

  class Model
  {
  private:
    xt::xarray<double> (*modelFunction)(xt::xarray<double>);
    std::string serverAddress;
    std::string modelName;
    std::string systemName;

  public:
    Model(xt::xarray<double> (*modelFunction)(xt::xarray<double>), const std::string& modelName = "Unnamed cpproblight C++ model");
    void startServer(const std::string& serverAddress = "tcp://*:5555");
  };

  xt::xarray<double> sample(distributions::Distribution& distribution);
  xt::xarray<double> sample(distributions::Distribution& distribution, const std::string& address);
  xt::xarray<double> sample(distributions::Distribution& distribution, const bool control, const bool replace);
  xt::xarray<double> sample(distributions::Distribution& distribution, const bool control, const bool replace, const std::string& address);
  void observe(distributions::Distribution& distribution, xt::xarray<double> value, const std::string& address="");

  void setDefaultControl(bool control = true);
  void setDefaultReplace(bool replace = false);

  xt::xarray<double> ProtocolTensorToXTensor(const PPLProtocol::ProtocolTensor* protocolTensor);

  flatbuffers::Offset<PPLProtocol::ProtocolTensor> XTensorToProtocolTensor(flatbuffers::FlatBufferBuilder& builder, xt::xarray<double> xtensor);

  void sendMessage(flatbuffers::Offset<PPLProtocol::Message> message);

  std::string demangleAddress(std::string address);

  std::string extractAddress();
}

#endif
