cmake_minimum_required(VERSION 3.1)

project(pyprob_cpp VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

add_subdirectory(pyprob_cpp)
include_directories(pyprob_cpp/include)

find_package(ZeroMQ REQUIRED)
find_package(FlatBuffers REQUIRED)
find_package(xtensor REQUIRED)
