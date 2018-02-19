#include "cpproblight.h"
#include <stdio.h>
#include <iostream>
#include "xtensor/xadapt.hpp"
// #include <unistd.h>
#include <typeinfo>
#include <cxxabi.h>
#include <cstdlib>


namespace cpproblight
{
  zmq::context_t zmqContext = zmq::context_t(1);
  zmq::socket_t zmqSocket = zmq::socket_t(zmqContext, ZMQ_REP);
  flatbuffers::FlatBufferBuilder builder;

  namespace distributions
  {
    double Distribution::sample(const bool control, const bool record_last_only, const std::string& address){
      return 0;
    }

    Uniform::Uniform(double low, double high)
    {
      this->low = low;
      this->high = high;
    }
    double Uniform::sample(const bool control, const bool record_last_only, const std::string& address)
    {
      auto uniform = PPLProtocol::CreateUniform(builder, this->low, this->high);
      auto addr = builder.CreateString(address);
      auto sample = PPLProtocol::CreateSample(builder, addr, PPLProtocol::Distribution_Uniform, uniform.Union());
      auto message_request = PPLProtocol::CreateMessage(builder, PPLProtocol::MessageBody_Sample, sample.Union());
      sendMessage(message_request);

      auto message_reply = receiveMessage();
      if (message_reply->body_type() == PPLProtocol::MessageBody_SampleResult)
      {
        printf("Protocol: Sample result received.\n");
        auto result = ProtocolTensorToXTensor(message_reply->body_as_SampleResult()->result());
        std::cout << result << std::endl;
        return result(0);
      }
      else
      {
        printf("Error: protocol received an unexpected request.\n");
        std::exit(EXIT_FAILURE);
      }
    }

    Normal::Normal(double mean, double stddev)
    {
      this->mean = mean;
      this->stddev = stddev;
    }
    double Normal::sample(const bool control, const bool record_last_only, const std::string& address)
    {
      return 0;
    }
  }

  Model::Model(xt::xarray<double> (*modelFunction)(xt::xarray<double>))
  {
    this->modelFunction = modelFunction;
  }

  void Model::startServer(const std::string& serverAddress)
  {
    this->serverAddress = serverAddress;

    zmqSocket.bind(serverAddress.c_str());
    printf("Protocol: ZMQ_REP server running at %s\n", this->serverAddress.c_str());

    while(true)
    {
      auto message = receiveMessage();
      if (message->body_type() == PPLProtocol::MessageBody_Run)
      {
        printf("Protocol: Run request received.\n");
        auto observation = ProtocolTensorToXTensor(message->body_as_Run()->observation());
        auto result = XTensorToProtocolTensor(builder, this->modelFunction(observation));

        auto runResult = PPLProtocol::CreateRunResult(builder, result);
        auto message = PPLProtocol::CreateMessage(builder, PPLProtocol::MessageBody_RunResult, runResult.Union());
        sendMessage(message);
      }
      else
      {
        printf("Error: protocol received an unexpected request.\n");
        std::exit(EXIT_FAILURE);
      }
    }
  }

  double sample(distributions::Distribution& distribution, const bool control, const bool record_last_only, const std::string& address)
  {
    return distribution.sample(control, record_last_only, address);
  }

  xt::xarray<double> ProtocolTensorToXTensor(const PPLProtocol::ProtocolTensor* protocolTensor)
  {
    auto data_begin = protocolTensor->data()->begin();
    auto data_size = protocolTensor->data()->size();
    auto data = std::vector<double>(data_begin, data_begin + data_size);

    auto shape_begin = protocolTensor->shape()->begin();
    auto shape_size = protocolTensor->shape()->size();
    auto shape = std::vector<int32_t>(shape_begin, shape_begin + shape_size);

    return xt::adapt(data, shape);
  }

  flatbuffers::Offset<PPLProtocol::ProtocolTensor> XTensorToProtocolTensor(flatbuffers::FlatBufferBuilder& builder, xt::xarray<double> xtensor)
  {
    auto shape = std::vector<int32_t>(xtensor.shape().begin(), xtensor.shape().end());
    auto data = std::vector<double>(xtensor.data().begin(), xtensor.data().end());
    return PPLProtocol::CreateProtocolTensorDirect(builder, &data, &shape);
  }

  const PPLProtocol::Message* receiveMessage()
  {
    zmq::message_t request;
    zmqSocket.recv(&request);

    return PPLProtocol::GetMessage(request.data());
  }

  void sendMessage(flatbuffers::Offset<PPLProtocol::Message> message)
  {
    builder.Finish(message);
    zmq::message_t request = zmq::message_t(builder.GetSize());
    memcpy(request.data(), builder.GetBufferPointer(), builder.GetSize());
    zmqSocket.send(request);
    builder.Clear();
  }
}
