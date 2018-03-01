#include "cpproblight.h"
#include <stdio.h>
#include <iostream>
#include "xtensor/xadapt.hpp"
#include <typeinfo>
#include <cxxabi.h>
#include <cstdlib>
#include <locale.h>
#include <execinfo.h>
#include <random>


namespace cpproblight
{
  zmq::context_t zmqContext = zmq::context_t(1);
  zmq::socket_t zmqSocket = zmq::socket_t(zmqContext, ZMQ_REP);
  bool zmqSocketConnected = false;
  flatbuffers::FlatBufferBuilder builder;

  namespace distributions
  {
    xt::xarray<double> Distribution::sample(const bool control, const bool replace, const std::string& address)
    {
      return xt::xarray<double> {0};
    }
    void Distribution::observe(xt::xarray<double> value)
    {
      return;
    }

    Uniform::Uniform(double low, double high)
    {
      this->low = low;
      this->high = high;
    }
    xt::xarray<double> Uniform::sample(const bool control, const bool replace, const std::string& address)
    {
      if (!zmqSocketConnected)
      {
        printf("Protocol (C++): Warning: Not connected, sampling locally.\n");
        std::default_random_engine generator;
        auto res = std::uniform_real_distribution<double>(this->low, this->high)(generator);
        return res;
      }
      auto uniform = PPLProtocol::CreateUniform(builder, this->low, this->high);
      auto sample = PPLProtocol::CreateSampleDirect(builder, address.c_str(), PPLProtocol::Distribution_Uniform, uniform.Union(), control, replace);
      auto message_request = PPLProtocol::CreateMessage(builder, PPLProtocol::MessageBody_Sample, sample.Union());
      sendMessage(message_request);

      auto message_reply = receiveMessage();
      if (message_reply->body_type() == PPLProtocol::MessageBody_SampleResult)
      {
        auto result = ProtocolTensorToXTensor(message_reply->body_as_SampleResult()->result());
        return result;
      }
      else
      {
        printf("Error: protocol received an unexpected request.\n");
        std::exit(EXIT_FAILURE);
      }
    }
    void Uniform::observe(xt::xarray<double> value)
    {
      if (!zmqSocketConnected)
      {
        printf("Protocol (C++): Warning: Not connected, observing locally.\n");
        return;
      }
      auto val = XTensorToProtocolTensor(builder, value);
      auto uniform = PPLProtocol::CreateUniform(builder, this->low, this->high);
      auto observe = PPLProtocol::CreateObserve(builder, PPLProtocol::Distribution_Uniform, uniform.Union(), val);
      auto message_request = PPLProtocol::CreateMessage(builder, PPLProtocol::MessageBody_Observe, observe.Union());
      sendMessage(message_request);

      auto message_reply = receiveMessage();
      return;
    }

    Normal::Normal(xt::xarray<double> mean, xt::xarray<double> stddev)
    {
      this->mean = mean;
      this->stddev = stddev;
    }
    xt::xarray<double> Normal::sample(const bool control, const bool replace, const std::string& address)
    {
      if (!zmqSocketConnected)
      {
        printf("Protocol (C++): Warning: Not connected, sampling locally.\n");
        std::default_random_engine generator;
        auto n = this->mean.size();
        xt::xarray<double> res = xt::xarray<double>(n);
        for (size_t i = 0; i < n; i++)
        {
          auto mean = this->mean(i);
          auto stddev = this->stddev(i);
          res(i) = std::normal_distribution<double>(mean, stddev)(generator);
        }
        return res;
      }
      auto mean = XTensorToProtocolTensor(builder, this->mean);
      auto stddev = XTensorToProtocolTensor(builder, this->stddev);
      auto normal = PPLProtocol::CreateNormal(builder, mean, stddev);
      auto sample = PPLProtocol::CreateSampleDirect(builder, address.c_str(), PPLProtocol::Distribution_Normal, normal.Union(), control, replace);
      auto message_request = PPLProtocol::CreateMessage(builder, PPLProtocol::MessageBody_Sample, sample.Union());
      sendMessage(message_request);

      auto message_reply = receiveMessage();
      if (message_reply->body_type() == PPLProtocol::MessageBody_SampleResult)
      {
        auto result = ProtocolTensorToXTensor(message_reply->body_as_SampleResult()->result());
        return result;
      }
      else
      {
        printf("Error: protocol received an unexpected request.\n");
        std::exit(EXIT_FAILURE);
      }
    }
    void Normal::observe(xt::xarray<double> value)
    {
      if (!zmqSocketConnected)
      {
        printf("Protocol (C++): Warning: Not connected, observing locally.\n");
        return;
      }
      auto val = XTensorToProtocolTensor(builder, value);
      auto mean = XTensorToProtocolTensor(builder, xt::xarray<double> {this->mean});
      auto stddev = XTensorToProtocolTensor(builder, xt::xarray<double> {this->stddev});
      auto normal = PPLProtocol::CreateNormal(builder, mean, stddev);
      auto observe = PPLProtocol::CreateObserve(builder, PPLProtocol::Distribution_Normal, normal.Union(), val);
      auto message_request = PPLProtocol::CreateMessage(builder, PPLProtocol::MessageBody_Observe, observe.Union());
      sendMessage(message_request);

      auto message_reply = receiveMessage();
      return;
    }
  }

  Model::Model(xt::xarray<double> (*modelFunction)(xt::xarray<double>), const std::string& modelName)
  {
    this->modelFunction = modelFunction;
    this->modelName = modelName;
    setlocale(LC_ALL,"");
    std::stringstream s;
    s << "cpproblight " << VERSION << " (" << GIT_BRANCH << ":" << GIT_COMMIT_HASH << ")";
    this->systemName = s.str();
  }

  void Model::startServer(const std::string& serverAddress)
  {
    this->serverAddress = serverAddress;
    zmqSocket.bind(serverAddress.c_str());
    zmqSocketConnected = true;
    printf("Protocol (C++): ZMQ_REP server listening at %s\n", this->serverAddress.c_str());
    printf("Protocol (C++): this system: %s\n", this->systemName.c_str());
    printf("Protocol (C++): model name : %s\n", this->modelName.c_str());

    int traces = 0;
    while(true)
    {
      auto message = receiveMessage();
      if (message->body_type() == PPLProtocol::MessageBody_Run)
      {
        printf("Executed traces: %'d\r", ++traces);
        std::cout.flush();

        auto observation = ProtocolTensorToXTensor(message->body_as_Run()->observation());
        auto result = XTensorToProtocolTensor(builder, this->modelFunction(observation));

        auto runResult = PPLProtocol::CreateRunResult(builder, result);
        auto message = PPLProtocol::CreateMessage(builder, PPLProtocol::MessageBody_RunResult, runResult.Union());
        sendMessage(message);
      }
      else if (message->body_type() == PPLProtocol::MessageBody_Handshake)
      {
        auto systemName = message->body_as_Handshake()->system_name()->str();
        printf("Protocol (C++): connected to PPL system: %s\n", systemName.c_str());
        auto handshakeResult = PPLProtocol::CreateHandshakeResultDirect(builder, this->systemName.c_str(), this->modelName.c_str());
        auto message = PPLProtocol::CreateMessage(builder, PPLProtocol::MessageBody_HandshakeResult, handshakeResult.Union());
        sendMessage(message);
      }
      else
      {
        printf("Error: protocol received an unexpected request.\n");
        std::exit(EXIT_FAILURE);
      }
    }
  }

  xt::xarray<double> sample(distributions::Distribution& distribution, const bool control, const bool replace, const std::string& address)
  {
    if (address.length() == 0)
    {
      return distribution.sample(control, replace, extractAddress());
    }
    else
    {
      return distribution.sample(control, replace, address);
    }
  }

  void observe(distributions::Distribution& distribution, xt::xarray<double> value)
  {
    return distribution.observe(value);
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

  std::string demangle(std::string str)
  {
    auto first = str.find_last_of('(') + 1;
    auto last = str.find_last_of(')');
    auto plus = str.find_last_of('+');

    int status = -1;
    char *result = abi::__cxa_demangle(str.substr(first, plus - first).c_str(), nullptr, nullptr, &status);
    if ( status == 0 )
    {
      auto demangled = std::string(result);
      std::free(result);
      return demangled + str.substr(plus, last - plus);
    }
    else
    {
      return str;
    }
  }

  std::string extractAddress()
  {
    const int buffer_size = 128;
    void *buffer[buffer_size];
    char **symbols;

    int nptrs = backtrace(buffer, buffer_size);
    if (nptrs == buffer_size)
    {
      printf("Warning: backtrace buffer full, addresses are likely to be truncated.\n");
    }
    symbols = backtrace_symbols(buffer, nptrs);
    if (symbols == nullptr)
    {
      std::perror("backtrace_symbols");
      std::exit(EXIT_FAILURE);
    }
    std::stringstream ret;
    ret << "[";
    bool begin = false;
    for (int j = nptrs - 1; j > 1; j--)
    {
      auto s = std::string(symbols[j]);
      if (begin)
      {
        ret << demangle(s);
        if (j != 2)
        {
          ret << "; ";
        }
      }
      if ((s.find("cpproblight") != std::string::npos) && (s.find("startServer") != std::string::npos))
      {
        begin = true;
      }
    }

    ret << "]";
    std::free(symbols);
    return ret.str();
  }

}
