#include "pyprob_cpp.h"
#include <stdio.h>
#include <iostream>
#include "xtensor/xadapt.hpp"
#include <typeinfo>
#include <cxxabi.h>
#include <cstdlib>
#include <locale.h>
#include <execinfo.h>


namespace pyprob_cpp
{
  zmq::context_t zmqContext = zmq::context_t(1);
  zmq::socket_t zmqSocket = zmq::socket_t(zmqContext, ZMQ_REP);
  bool zmqSocketConnected = false;
  flatbuffers::FlatBufferBuilder builder;
  bool defaultControl = true;
  bool defaultReplace = false;

  namespace distributions
  {
    xt::xarray<double> Distribution::sample(const bool control, const bool replace, const std::string& address)
    {
      return xt::xarray<double> {0};
    }
    void Distribution::observe(xt::xarray<double> value, const std::string& address)
    {
      return;
    }

    Uniform::Uniform(xt::xarray<double> low, xt::xarray<double> high)
    {
      this->low = low;
      this->high = high;
    }
    xt::xarray<double> Uniform::sample(const bool control, const bool replace, const std::string& address)
    {

      if (!zmqSocketConnected)
      {
        printf("PPX (C++): Warning: Not connected, sampling locally.\n");
        auto n = this->low.size();
        xt::xtensor<double, 1> res(std::array<size_t, 1>{n});
        for (size_t i = 0; i < n; i++)
        {
          auto low = this->low(i);
          auto high = this->high(i);
          res(i) = std::uniform_real_distribution<double>(low, high)(generator);
        }
        return res;
      }
      auto low = XTensorToTensor(builder, this->low);
      auto high = XTensorToTensor(builder, this->high);
      auto uniform = ppx::CreateUniform(builder, low, high);
      auto sample = ppx::CreateSampleDirect(builder, address.c_str(), ppx::Distribution_Uniform, uniform.Union(), control, replace);
      auto message_request = ppx::CreateMessage(builder, ppx::MessageBody_Sample, sample.Union());
      sendMessage(message_request);

      zmq::message_t request;
      zmqSocket.recv(&request);
      auto message_reply = ppx::GetMessage(request.data());
      if (message_reply->body_type() == ppx::MessageBody_SampleResult)
      {
        auto result = TensorToXTensor(message_reply->body_as_SampleResult()->result());
        return result;
      }
      else
      {
        printf("PPX (C++): Error: Received an unexpected request. Cannot recover.\n");
        std::exit(EXIT_FAILURE);
      }
    }
    void Uniform::observe(xt::xarray<double> value, const std::string& address)
    {
      if (!zmqSocketConnected)
      {
        printf("PPX (C++): Warning: Not connected, observing locally.\n");
        return;
      }
      auto val = XTensorToTensor(builder, value);
      auto low = XTensorToTensor(builder, this->low);
      auto high = XTensorToTensor(builder, this->high);
      auto uniform = ppx::CreateUniform(builder, low, high);
      auto observe = ppx::CreateObserveDirect(builder, address.c_str(), ppx::Distribution_Uniform, uniform.Union(), val);
      auto message_request = ppx::CreateMessage(builder, ppx::MessageBody_Observe, observe.Union());
      sendMessage(message_request);

      zmq::message_t request;
      zmqSocket.recv(&request);
      // auto message_reply = ppx::GetMessage(request.data());
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
        printf("PPX (C++): Warning: Not connected, sampling locally.\n");
        auto n = this->mean.size();
        xt::xtensor<double, 1> res(std::array<size_t, 1>{n});
        for (size_t i = 0; i < n; i++)
        {
          auto mean = this->mean(i);
          auto stddev = this->stddev(i);
          res(i) = std::normal_distribution<double>(mean, stddev)(generator);
        }
        return res;
      }
      auto mean = XTensorToTensor(builder, this->mean);
      auto stddev = XTensorToTensor(builder, this->stddev);
      auto normal = ppx::CreateNormal(builder, mean, stddev);
      auto sample = ppx::CreateSampleDirect(builder, address.c_str(), ppx::Distribution_Normal, normal.Union(), control, replace);
      auto message_request = ppx::CreateMessage(builder, ppx::MessageBody_Sample, sample.Union());
      sendMessage(message_request);

      zmq::message_t request;
      zmqSocket.recv(&request);
      auto message_reply = ppx::GetMessage(request.data());
      if (message_reply->body_type() == ppx::MessageBody_SampleResult)
      {
        auto result = TensorToXTensor(message_reply->body_as_SampleResult()->result());
        return result;
      }
      else
      {
        printf("PPX (C++): Error: Received an unexpected request. Cannot recover.\n");
        std::exit(EXIT_FAILURE);
      }
    }
    void Normal::observe(xt::xarray<double> value, const std::string& address)
    {
      if (!zmqSocketConnected)
      {
        printf("PPX (C++): Warning: Not connected, observing locally.\n");
        return;
      }
      auto val = XTensorToTensor(builder, value);
      auto mean = XTensorToTensor(builder, this->mean);
      auto stddev = XTensorToTensor(builder, this->stddev);
      auto normal = ppx::CreateNormal(builder, mean, stddev);
      auto observe = ppx::CreateObserveDirect(builder, address.c_str(), ppx::Distribution_Normal, normal.Union(), val);
      auto message_request = ppx::CreateMessage(builder, ppx::MessageBody_Observe, observe.Union());
      sendMessage(message_request);

      zmq::message_t request;
      zmqSocket.recv(&request);
      // auto message_reply = ppx::GetMessage(request.data());
      return;
    }

    Categorical::Categorical(xt::xarray<double> probs)
    {
      this->probs = probs;
    }
    xt::xarray<double> Categorical::sample(const bool control, const bool replace, const std::string& address)
    {
      if (!zmqSocketConnected)
      {
        printf("PPX (C++): Warning: Not connected, sampling locally.\n");
        auto res = std::discrete_distribution<int>(this->probs.data().begin(), this->probs.data().end())(generator);
        return res;
      }
      auto probs = XTensorToTensor(builder, this->probs);
      auto categorical = ppx::CreateCategorical(builder, probs);
      auto sample = ppx::CreateSampleDirect(builder, address.c_str(), ppx::Distribution_Categorical, categorical.Union(), control, replace);
      auto message_request = ppx::CreateMessage(builder, ppx::MessageBody_Sample, sample.Union());
      sendMessage(message_request);

      zmq::message_t request;
      zmqSocket.recv(&request);
      auto message_reply = ppx::GetMessage(request.data());
      if (message_reply->body_type() == ppx::MessageBody_SampleResult)
      {
        auto result = TensorToXTensor(message_reply->body_as_SampleResult()->result());
        return result;
      }
      else
      {
        printf("PPX (C++): Error: Received an unexpected request. Cannot recover.\n");
        std::exit(EXIT_FAILURE);
      }
    }
    void Categorical::observe(xt::xarray<double> value, const std::string& address)
    {
      if (!zmqSocketConnected)
      {
        printf("PPX (C++): Warning: Not connected, observing locally.\n");
        return;
      }
      auto val = XTensorToTensor(builder, value);
      auto probs = XTensorToTensor(builder, this->probs);
      auto categorical = ppx::CreateCategorical(builder, probs);
      auto observe = ppx::CreateObserveDirect(builder, address.c_str(), ppx::Distribution_Categorical, categorical.Union(), val);
      auto message_request = ppx::CreateMessage(builder, ppx::MessageBody_Observe, observe.Union());
      sendMessage(message_request);

      zmq::message_t request;
      zmqSocket.recv(&request);
      // auto message_reply = ppx::GetMessage(request.data());
      return;
    }

    Poisson::Poisson(xt::xarray<double> rate)
    {
      this->rate = rate;
    }
    xt::xarray<double> Poisson::sample(const bool control, const bool replace, const std::string& address)
    {

      if (!zmqSocketConnected)
      {
        printf("PPX (C++): Warning: Not connected, sampling locally.\n");
        auto n = this->rate.size();
        xt::xtensor<double, 1> res(std::array<size_t, 1>{n});
        for (size_t i = 0; i < n; i++)
        {
          auto rate = this->rate(i);
          res(i) = std::poisson_distribution<int>(rate)(generator);
        }
        return res;
      }
      auto rate = XTensorToTensor(builder, this->rate);
      auto poisson = ppx::CreatePoisson(builder, rate);
      auto sample = ppx::CreateSampleDirect(builder, address.c_str(), ppx::Distribution_Poisson, poisson.Union(), control, replace);
      auto message_request = ppx::CreateMessage(builder, ppx::MessageBody_Sample, sample.Union());
      sendMessage(message_request);

      zmq::message_t request;
      zmqSocket.recv(&request);
      auto message_reply = ppx::GetMessage(request.data());
      if (message_reply->body_type() == ppx::MessageBody_SampleResult)
      {
        auto result = TensorToXTensor(message_reply->body_as_SampleResult()->result());
        return result;
      }
      else
      {
        printf("PPX (C++): Error: Received an unexpected request. Cannot recover.\n");
        std::exit(EXIT_FAILURE);
      }
    }
    void Poisson::observe(xt::xarray<double> value, const std::string& address)
    {
      if (!zmqSocketConnected)
      {
        printf("PPX (C++): Warning: Not connected, observing locally.\n");
        return;
      }
      auto val = XTensorToTensor(builder, value);
      auto rate = XTensorToTensor(builder, this->rate);
      auto poisson = ppx::CreatePoisson(builder, rate);
      auto observe = ppx::CreateObserveDirect(builder, address.c_str(), ppx::Distribution_Poisson, poisson.Union(), val);
      auto message_request = ppx::CreateMessage(builder, ppx::MessageBody_Observe, observe.Union());
      sendMessage(message_request);

      zmq::message_t request;
      zmqSocket.recv(&request);
      // auto message_reply = ppx::GetMessage(request.data());
      return;
    }
  }

  Model::Model(xt::xarray<double> (*modelFunction)(xt::xarray<double>), xt::xarray<double> defaultObservation, const std::string& modelName)
  {
    this->modelFunction = modelFunction;
    this->defaultObservation = defaultObservation;
    this->modelName = modelName;
    setlocale(LC_ALL,"");
    std::stringstream s;
    s << "pyprob_cpp " << VERSION << " (" << GIT_BRANCH << ":" << GIT_COMMIT_HASH << ")";
    this->systemName = s.str();
  }

  void Model::startServer(const std::string& serverAddress)
  {
    this->serverAddress = serverAddress;
    zmqSocket.bind(serverAddress.c_str());
    zmqSocketConnected = true;
    printf("PPX (C++): ZMQ_REP server listening at %s\n", this->serverAddress.c_str());
    printf("PPX (C++): This system: %s\n", this->systemName.c_str());
    printf("PPX (C++): Model name : %s\n", this->modelName.c_str());

    int traces = 0;
    while(true)
    {
      zmq::message_t request;
      zmqSocket.recv(&request);
      auto message = ppx::GetMessage(request.data());
      if (message->body_type() == ppx::MessageBody_Run)
      {
        printf("PPX (C++): Executed traces: %'d\r", ++traces);
        std::cout.flush();

        xt::xarray<double> obs;
        if (message->body_as_Run()->observation() == NULL)
        {
          obs = this->defaultObservation;
        }
        else
        {
          obs = TensorToXTensor(message->body_as_Run()->observation());
        }
        auto result = XTensorToTensor(builder, this->modelFunction(obs));

        auto runResult = ppx::CreateRunResult(builder, result);
        auto message = ppx::CreateMessage(builder, ppx::MessageBody_RunResult, runResult.Union());
        sendMessage(message);
      }
      else if (message->body_type() == ppx::MessageBody_Handshake)
      {
        auto systemName = message->body_as_Handshake()->system_name()->str();
        printf("PPX (C++): Connected to PPL system: %s\n", systemName.c_str());
        auto handshakeResult = ppx::CreateHandshakeResultDirect(builder, this->systemName.c_str(), this->modelName.c_str());
        auto message = ppx::CreateMessage(builder, ppx::MessageBody_HandshakeResult, handshakeResult.Union());
        sendMessage(message);
      }
      else
      {
        printf("PPX (C++): Error: Received an unexpected request. Resetting...\n");
        auto reset = ppx::CreateReset(builder);
        auto message = ppx::CreateMessage(builder, ppx::MessageBody_Reset, reset.Union());
        sendMessage(message);
      }
    }
  }

  xt::xarray<double> sample(distributions::Distribution& distribution)
  {
    auto address = extractAddress();
    return distribution.sample(defaultControl, defaultReplace, address);
  }

  xt::xarray<double> sample(distributions::Distribution& distribution, const std::string& address)
  {
    return distribution.sample(defaultControl, defaultReplace, address);
  }

  xt::xarray<double> sample(distributions::Distribution& distribution, const bool control, const bool replace)
  {
    auto address = extractAddress();
    return distribution.sample(control, replace, address);
  }

  xt::xarray<double> sample(distributions::Distribution& distribution, const bool control, const bool replace, const std::string& address)
  {
    return distribution.sample(control, replace, address);
  }

  void observe(distributions::Distribution& distribution, xt::xarray<double> value, const std::string& address)
  {
    auto addr = address;
    if (addr.length() == 0)
    {
      addr = extractAddress();
    }
    return distribution.observe(value, addr);
  }

  void setDefaultControl(bool control)
  {
    defaultControl = control;
  }

  void setDefaultReplace(bool replace)
  {
    defaultReplace = replace;
  }

  xt::xarray<double> TensorToXTensor(const ppx::Tensor* protocolTensor)
  {
    auto data_begin = protocolTensor->data()->begin();
    auto data_size = protocolTensor->data()->size();
    auto data = std::vector<double>(data_begin, data_begin + data_size);

    auto shape_begin = protocolTensor->shape()->begin();
    auto shape_size = protocolTensor->shape()->size();
    auto shape = std::vector<int32_t>(shape_begin, shape_begin + shape_size);

    return xt::adapt(data, shape);
  }

  flatbuffers::Offset<ppx::Tensor> XTensorToTensor(flatbuffers::FlatBufferBuilder& builder, xt::xarray<double> xtensor)
  {
    auto shape = std::vector<int32_t>(xtensor.shape().begin(), xtensor.shape().end());
    auto data = std::vector<double>(xtensor.data().begin(), xtensor.data().end());
    return ppx::CreateTensorDirect(builder, &data, &shape);
  }

  void sendMessage(flatbuffers::Offset<ppx::Message> message)
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
      if ((s.find("pyprob_cpp") != std::string::npos) && (s.find("startServer") != std::string::npos))
      {
        begin = true;
      }
    }

    ret << "]";
    std::free(symbols);
    return ret.str();
  }

}
