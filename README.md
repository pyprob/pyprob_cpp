# pyprob_cpp [![Build Status](https://travis-ci.org/probprog/pyprob_cpp.svg?branch=master)](https://travis-ci.org/probprog/pyprob_cpp)

`pyprob_cpp` is a C++ library providing a lightweight interface to the `pyprob` probabilistic programming library implemented in Python. The two components communicate through a `pplprotocol` interface that allows execution of models and inference engines in separate programming languages, processes, and machines connected over a network. 

Please see the main [pyprob](https://github.com/probprog/pyprob) documentation for more information.

## Dependencies

- ZMQ: http://zeromq.org/
- flatbuffers: https://github.com/google/flatbuffers/releases
- xtensor: https://github.com/QuantStack/xtl

## Docker

A Docker image with the latest passing commit is automatically pushed to `probprog/pyprob_cpp:latest`

https://hub.docker.com/r/probprog/pyprob_cpp/

## License

`pyprob_cpp` is distributed under the BSD License.
