project(libcpproblight VERSION 0.1 LANGUAGES CXX)

add_library(cpproblight
  src/cpproblight.cpp)

target_include_directories(cpproblight PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

add_executable(test_gum test/test_gum.cpp)
target_link_libraries(test_gum cpproblight zmq)

add_test(test_gum_1 test_gum)
