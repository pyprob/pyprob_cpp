FROM ubuntu:18.04

ENV CC=gcc-5
ENV CXX=g++-5

RUN apt-get update && apt-get install -y git cmake gcc-5 g++-5 libzmq3-dev

WORKDIR /home
RUN git clone --branch v2.0.0 https://github.com/google/flatbuffers.git && cd flatbuffers && cmake -G "Unix Makefiles" && make install
RUN git clone --branch 0.6.13 https://github.com/QuantStack/xtl.git && cd xtl && cmake . && make install
RUN git clone --branch 0.21.4 https://github.com/QuantStack/xtensor.git && cd xtensor && cmake . && make install

COPY . ./pyprob_cpp
RUN cd pyprob_cpp && rm -rf build && mkdir build && cd build && cmake ../src && cmake --build . && make install

ARG GIT_COMMIT="unknown"

LABEL project="pyprob_cpp"
LABEL url="https://github.com/pyprob/pyprob_cpp"
LABEL maintainer="Atilim Gunes Baydin <gunes@robots.ox.ac.uk>"
LABEL git_commit=$GIT_COMMIT
