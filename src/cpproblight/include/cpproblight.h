#ifndef CPPROBLIGHT_H
#define CPPROBLIGHT_H
#include "version.h"
#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "pplprotocol_generated.h"
#include <zmq.hpp>


namespace cpproblight
{
  namespace distributions
  {
    class Distribution
    {
    public:
      virtual xt::xarray<double> sample(const bool control, const bool record_last_only, const std::string& address);
      virtual void observe(xt::xarray<double> value);
    };

    class Uniform: public Distribution
    {
    private:
      double low;
      double high;

    public:
      Uniform(double low=0, double high=1);
      xt::xarray<double> sample(const bool control, const bool record_last_only, const std::string& address);
      void observe(xt::xarray<double> value);
    };

    class Normal: public Distribution
    {
    private:
      double mean;
      double stddev;

    public:
      Normal(double mean=0, double stddev=1);
      xt::xarray<double> sample(const bool control, const bool record_last_only, const std::string& address);
      void observe(xt::xarray<double> value);
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

  xt::xarray<double> sample(distributions::Distribution& distribution, const bool control=true, const bool record_last_only=false, const std::string& address="");

  void observe(distributions::Distribution& distribution, xt::xarray<double> value);

  xt::xarray<double> ProtocolTensorToXTensor(const PPLProtocol::ProtocolTensor* protocolTensor);

  flatbuffers::Offset<PPLProtocol::ProtocolTensor> XTensorToProtocolTensor(flatbuffers::FlatBufferBuilder& builder, xt::xarray<double> xtensor);

  const PPLProtocol::Message* receiveMessage();
  void sendMessage(flatbuffers::Offset<PPLProtocol::Message> message);
}

#endif
