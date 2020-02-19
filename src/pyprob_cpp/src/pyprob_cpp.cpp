#include "pyprob_cpp.h"
#include <stdio.h>
#include <iostream>
#include "xtensor/xadapt.hpp"
#include <typeinfo>
#include <cxxabi.h>
#include <cstdlib>
#include <locale.h>
#include <execinfo.h>

#include <unordered_map>
#include <dlfcn.h>

namespace pyprob_cpp
{
  std::default_random_engine generator;
  zmq::context_t zmqContext = zmq::context_t(1);
  zmq::socket_t zmqSocket = zmq::socket_t(zmqContext, ZMQ_REP);
  bool zmqSocketConnected = false;
  flatbuffers::FlatBufferBuilder builder;
  bool defaultControl = true;
  bool defaultReplace = false;

  namespace distributions
  {
    xt::xarray<double> Distribution::sample(const bool control, const bool replace, const std::string& address, const std::string& name)
    {
      return xt::xarray<double> {0};
    }
    void Distribution::observe(xt::xarray<double> value, const std::string& address, const std::string& name)
    {
      return;
    }

    Uniform::Uniform(xt::xarray<double> low, xt::xarray<double> high)
    {
      this->low = low;
      this->high = high;
    }
    xt::xarray<double> Uniform::sample(const bool control, const bool replace, const std::string& address, const std::string& name)
    {

      if (!zmqSocketConnected)
      {
        printf("ppx (C++): Warning: Not connected, sampling locally.\n");
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
      auto sample = ppx::CreateSampleDirect(builder, address.c_str(), name.c_str(), ppx::Distribution_Uniform, uniform.Union(), control, replace);
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
        printf("ppx (C++): Error: Received an unexpected request. Cannot recover.\n");
        std::exit(EXIT_FAILURE);
      }
    }
    void Uniform::observe(xt::xarray<double> value, const std::string& address, const std::string& name)
    {
      if (!zmqSocketConnected)
      {
        printf("ppx (C++): Warning: Not connected, observing locally.\n");
        return;
      }
      flatbuffers::Offset<ppx::Tensor> val = 0;
      if (value(0) != NONE_VALUE)
        val = XTensorToTensor(builder, value);
      auto low = XTensorToTensor(builder, this->low);
      auto high = XTensorToTensor(builder, this->high);
      auto uniform = ppx::CreateUniform(builder, low, high);
      auto observe = ppx::CreateObserveDirect(builder, address.c_str(), name.c_str(), ppx::Distribution_Uniform, uniform.Union(), val);
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
    xt::xarray<double> Normal::sample(const bool control, const bool replace, const std::string& address, const std::string& name)
    {
      if (!zmqSocketConnected)
      {
        printf("ppx (C++): Warning: Not connected, sampling locally.\n");
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
      auto sample = ppx::CreateSampleDirect(builder, address.c_str(), name.c_str(), ppx::Distribution_Normal, normal.Union(), control, replace);
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
        printf("ppx (C++): Error: Received an unexpected request. Cannot recover.\n");
        std::exit(EXIT_FAILURE);
      }
    }
    void Normal::observe(xt::xarray<double> value, const std::string& address, const std::string& name)
    {
      if (!zmqSocketConnected)
      {
        printf("ppx (C++): Warning: Not connected, observing locally.\n");
        return;
      }
      flatbuffers::Offset<ppx::Tensor> val = 0;
      if (value(0) != NONE_VALUE)
        val = XTensorToTensor(builder, value);
      auto mean = XTensorToTensor(builder, this->mean);
      auto stddev = XTensorToTensor(builder, this->stddev);
      auto normal = ppx::CreateNormal(builder, mean, stddev);
      auto observe = ppx::CreateObserveDirect(builder, address.c_str(), name.c_str(), ppx::Distribution_Normal, normal.Union(), val);
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
    xt::xarray<double> Categorical::sample(const bool control, const bool replace, const std::string& address, const std::string& name)
    {
      if (!zmqSocketConnected)
      {
        printf("ppx (C++): Warning: Not connected, sampling locally.\n");
        auto res = std::discrete_distribution<int>(this->probs.storage().begin(), this->probs.storage().end())(generator);
        return res;
      }
      auto probs = XTensorToTensor(builder, this->probs);
      auto categorical = ppx::CreateCategorical(builder, probs);
      auto sample = ppx::CreateSampleDirect(builder, address.c_str(), name.c_str(), ppx::Distribution_Categorical, categorical.Union(), control, replace);
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
        printf("ppx (C++): Error: Received an unexpected request. Cannot recover.\n");
        std::exit(EXIT_FAILURE);
      }
    }
    void Categorical::observe(xt::xarray<double> value, const std::string& address, const std::string& name)
    {
      if (!zmqSocketConnected)
      {
        printf("ppx (C++): Warning: Not connected, observing locally.\n");
        return;
      }
      flatbuffers::Offset<ppx::Tensor> val = 0;
      if (value(0) != NONE_VALUE)
        val = XTensorToTensor(builder, value);
      auto probs = XTensorToTensor(builder, this->probs);
      auto categorical = ppx::CreateCategorical(builder, probs);
      auto observe = ppx::CreateObserveDirect(builder, address.c_str(), name.c_str(), ppx::Distribution_Categorical, categorical.Union(), val);
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
    xt::xarray<double> Poisson::sample(const bool control, const bool replace, const std::string& address, const std::string& name)
    {

      if (!zmqSocketConnected)
      {
        printf("ppx (C++): Warning: Not connected, sampling locally.\n");
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
      auto sample = ppx::CreateSampleDirect(builder, address.c_str(), name.c_str(), ppx::Distribution_Poisson, poisson.Union(), control, replace);
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
        printf("ppx (C++): Error: Received an unexpected request. Cannot recover.\n");
        std::exit(EXIT_FAILURE);
      }
    }
    void Poisson::observe(xt::xarray<double> value, const std::string& address, const std::string& name)
    {
      if (!zmqSocketConnected)
      {
        printf("ppx (C++): Warning: Not connected, observing locally.\n");
        return;
      }
      flatbuffers::Offset<ppx::Tensor> val = 0;
      if (value(0) != NONE_VALUE)
        val = XTensorToTensor(builder, value);
      auto rate = XTensorToTensor(builder, this->rate);
      auto poisson = ppx::CreatePoisson(builder, rate);
      auto observe = ppx::CreateObserveDirect(builder, address.c_str(), name.c_str(), ppx::Distribution_Poisson, poisson.Union(), val);
      auto message_request = ppx::CreateMessage(builder, ppx::MessageBody_Observe, observe.Union());
      sendMessage(message_request);

      zmq::message_t request;
      zmqSocket.recv(&request);
      // auto message_reply = ppx::GetMessage(request.data());
      return;
    }
  }

  Model::Model(xt::xarray<double> (*modelFunction)(), const std::string& modelName)
  {
    this->modelFunction = modelFunction;
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
    printf("ppx (C++): ZMQ_REP server listening at %s\n", this->serverAddress.c_str());
    printf("ppx (C++): This system: %s\n", this->systemName.c_str());
    printf("ppx (C++): Model name : %s\n", this->modelName.c_str());

    int traces = 0;
    while(true)
    {
      zmq::message_t request;
      zmqSocket.recv(&request);
      auto message = ppx::GetMessage(request.data());
      if (message->body_type() == ppx::MessageBody_Run)
      {
        printf("ppx (C++): Executed traces: %'d\r", ++traces);
        std::cout.flush();

        auto result = XTensorToTensor(builder, this->modelFunction());

        auto runResult = ppx::CreateRunResult(builder, result);
        auto message = ppx::CreateMessage(builder, ppx::MessageBody_RunResult, runResult.Union());
        sendMessage(message);
      }
      else if (message->body_type() == ppx::MessageBody_Handshake)
      {
        auto systemName = message->body_as_Handshake()->system_name()->str();
        printf("ppx (C++): Connected to PPL system: %s\n", systemName.c_str());
        auto handshakeResult = ppx::CreateHandshakeResultDirect(builder, this->systemName.c_str(), this->modelName.c_str());
        auto message = ppx::CreateMessage(builder, ppx::MessageBody_HandshakeResult, handshakeResult.Union());
        sendMessage(message);
      }
      else
      {
        printf("ppx (C++): Error: Received an unexpected request. Resetting...\n");
        auto reset = ppx::CreateReset(builder);
        auto message = ppx::CreateMessage(builder, ppx::MessageBody_Reset, reset.Union());
        sendMessage(message);
      }
    }
  }

  xt::xarray<double> sample(distributions::Distribution& distribution)
  {
    auto address = extractAddress();
    return distribution.sample(defaultControl, defaultReplace, address, "");
  }

  xt::xarray<double> sample(distributions::Distribution& distribution, const std::string& name)
  {
    auto address = extractAddress();
    return distribution.sample(defaultControl, defaultReplace, address, name);
  }

  xt::xarray<double> sample(distributions::Distribution& distribution, const bool control, const bool replace)
  {
    auto address = extractAddress();
    return distribution.sample(control, replace, address, "");
  }

  xt::xarray<double> sample(distributions::Distribution& distribution, const bool control, const bool replace, const std::string& name)
  {
    auto address = extractAddress();
    return distribution.sample(control, replace, address, name);
  }

  void observe(distributions::Distribution& distribution, xt::xarray<double> value)
  {
    auto address = extractAddress();
    return distribution.observe(value, address, "");
  }

  void observe(distributions::Distribution& distribution, const std::string& name)
  {
    auto address = extractAddress();
    return distribution.observe(NONE_VALUE, address, name);
  }

  void observe(distributions::Distribution& distribution, xt::xarray<double> value, const std::string& name)
  {
    auto address = extractAddress();
    return distribution.observe(value, address, name);
  }

  void tag(xt::xarray<double> value)
  {
    tag(value, "");
  }

  void tag(xt::xarray<double> value, const std::string& name)
  {
    auto address = extractAddress();
    if (!zmqSocketConnected)
    {
      printf("ppx (C++): Warning: Not connected, tagging locally.\n");
      return;
    }
    auto val = XTensorToTensor(builder, value);
    auto tag = ppx::CreateTagDirect(builder, address.c_str(), name.c_str(), val);
    auto message_request = ppx::CreateMessage(builder, ppx::MessageBody_Tag, tag.Union());
    sendMessage(message_request);

    zmq::message_t request;
    zmqSocket.recv(&request);
    // auto message_reply = ppx::GetMessage(request.data());
    return;
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
    auto data = std::vector<double>(xtensor.storage().begin(), xtensor.storage().end());
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
    static std::unordered_map<void *,std::string> addr2str;

    int nptrs = backtrace(buffer, buffer_size);
    if (nptrs == buffer_size)
    {
      printf("Warning: backtrace buffer full, addresses are likely to be truncated.\n");
    }

    std::stringstream ret;
    ret << "[";
    bool begin = false;
    for (int j = nptrs - 1; j > 1; j--)
    {
      const std::string *s = 0;
      auto it = addr2str.find(buffer[j]);
      if (it != addr2str.end())
	s = &it->second;
      else
      {
	Dl_info info;
	int err = dladdr(buffer[j], &info);
	if (err == 0) {
	  std::perror("dladdr");
	  std::exit(EXIT_FAILURE);
	}
	int status;
	const char *dpp;
	char *dp = abi::__cxa_demangle(info.dli_sname, 0, 0, &status);
	if (status != 0)
	  dpp = info.dli_sname;
	else
	  dpp = dp;
	char namebuf[1024];
	snprintf(namebuf, sizeof(namebuf), "%s+0x%lx", dpp,
	    (uint64_t)buffer[j] - (uint64_t)info.dli_saddr);
	if (dp)
	  free(dp);
	auto ret = addr2str.insert(std::pair<void *,std::string>(
				buffer[j],namebuf));
	s = &ret.first->second;
      }

      if (begin)
      {
        ret << *s;
        if (j != 2)
        {
          ret << "; ";
        }
      }
      if ((s->find("pyprob_cpp") != std::string::npos) && (s->find("startServer") != std::string::npos))
      {
        begin = true;
      }
    }

    ret << "]";
    return ret.str();
  }

}
