FROM ubuntu:16.04

ENV CC=gcc-5
ENV CXX=g++-5

RUN apt-get update
RUN apt-get install -y git cmake gcc-5 g++-5 libzmq3-dev
RUN git clone --branch v1.8.0 https://github.com/google/flatbuffers.git && cd flatbuffers && cmake -G "Unix Makefiles" && make install && cd ..
RUN git clone --branch 0.4.0 https://github.com/QuantStack/xtl.git && cd xtl && cmake . && make install && cd ..
RUN git clone --branch 0.15.4 https://github.com/QuantStack/xtensor.git && cd xtensor && cmake . && make install && cd ..

COPY ./src /code/cpproblight/src
RUN cd /code/cpproblight && mkdir build && cd build && cmake ../src && cmake --build .

ARG GIT_COMMIT="unknown"

LABEL project="cpproblight"
LABEL url="https://github.com/probprog/cpproblight"
LABEL maintainer="Atilim Gunes Baydin <gunes@robots.ox.ac.uk>"
LABEL git_commit=$GIT_COMMIT

WORKDIR /workspace
RUN chmod -R a+w /workspace
