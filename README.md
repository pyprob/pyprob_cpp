# pyprob_cpp [![Build Status](https://travis-ci.org/probprog/pyprob_cpp.svg?branch=master)](https://travis-ci.org/probprog/pyprob_cpp)

`pyprob_cpp` is a C++ library providing a lightweight interface to the `pyprob` probabilistic programming library implemented in Python. The two components communicate through the [PPX](https://github.com/probprog/ppx) protocol that allows execution of models and inference engines in separate programming languages, processes, and machines connected over a network. 

Please see the main [pyprob](https://github.com/probprog/pyprob) documentation for more information.

## Installation

### Dependencies
- ZMQ: http://zeromq.org/
- flatbuffers: https://github.com/google/flatbuffers/releases
- xtensor: https://github.com/QuantStack/xtl

### Install from source

Please see the provided [Dockerfile](https://github.com/probprog/pyprob_cpp/blob/master/Dockerfile) for more specific instructions on how to install the dependencies and configure the build environment.

```
git clone git@github.com:probprog/pyprob_cpp.git
cd pyprob_cpp
mkdir build && cd build && cmake ../src && cmake --build . && make install
```

## Docker

A Docker image with the latest passing commit is automatically pushed to `probprog/pyprob_cpp:latest`

https://hub.docker.com/r/probprog/pyprob_cpp/

## Example models

Several example models in C++ are provided in this repository, under the [src/pyprob_cpp/test](https://github.com/probprog/pyprob_cpp/tree/master/src/pyprob_cpp/test) folder. 

These mirror the Python test cases in the main `pyprob` repository and are used as continuous integration tests ensuring that `pyprob` and `pyprob_cpp` work seamlessly together.

Here is how a simple model looks like:

```cpp
#include <pyprob_cpp.h>

// Gaussian with unkown mean
// http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf

xt::xarray<double> forward(xt::xarray<double> observation)
{
  auto prior_mean = 1;
  auto prior_stddev = std::sqrt(5);
  auto likelihood_stddev = std::sqrt(2);

  auto prior = pyprob_cpp::distributions::Normal(prior_mean, prior_stddev);
  auto mu = pyprob_cpp::sample(prior);

  auto likelihood = pyprob_cpp::distributions::Normal(mu, likelihood_stddev);
  for (auto & o : observation)
  {
    pyprob_cpp::observe(likelihood, o);
  }

  return mu;
}


int main(int argc, char *argv[])
{
  auto serverAddress = (argc > 1) ? argv[1] : "tcp://*:5555";
  pyprob_cpp::Model model = pyprob_cpp::Model(forward, xt::xarray<double> {}, "Gaussian with unknown mean C++");
  model.startServer(serverAddress);
  return 0;
}
```

## License

`pyprob_cpp` is distributed under the BSD License.
