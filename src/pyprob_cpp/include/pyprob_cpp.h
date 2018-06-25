#ifndef pyprob_cpp_H
#define pyprob_cpp_H
#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "pplprotocol.h"
#include <zmq.hpp>
#include <random>

#define VERSION "0.1.3"
#define GIT_BRANCH "master"
#define GIT_COMMIT_HASH "cf5565c"


namespace pyprob_cpp
{
  std::default_random_engine generator;
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
      xt::xarray<double> low;
      xt::xarray<double> high;

    public:
      Uniform(xt::xarray<double> low=xt::xarray<double> {0}, xt::xarray<double> high=xt::xarray<double> {1});
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

    class Categorical: public Distribution
    {
    private:
      xt::xarray<double> probs;

    public:
      Categorical(xt::xarray<double> probs=xt::xarray<double> {1});
      xt::xarray<double> sample(const bool control, const bool replace, const std::string& address);
      void observe(xt::xarray<double> value, const std::string& address);
    };

    class Poisson: public Distribution
    {
    private:
      xt::xarray<double> rate;

    public:
      Poisson(xt::xarray<double> rate=0);
      xt::xarray<double> sample(const bool control, const bool replace, const std::string& address);
      void observe(xt::xarray<double> value, const std::string& address);
    };
  }

  class Model
  {
  private:
    xt::xarray<double> (*modelFunction)(xt::xarray<double>);
    xt::xarray<double> defaultObservation;
    std::string serverAddress;
    std::string modelName;
    std::string systemName;

  public:
    Model(xt::xarray<double> (*modelFunction)(xt::xarray<double>), xt::xarray<double> defaultObservation, const std::string& modelName);
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
