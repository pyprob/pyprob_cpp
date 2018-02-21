project(libcpproblight VERSION 0.1 LANGUAGES CXX)

set (VERSION "0.1.0")

add_library(cpproblight
  src/cpproblight.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_gum test/test_gum.cpp)
target_link_libraries(test_gum cpproblight zmq)

add_executable(test_gum_marsaglia test/test_gum_marsaglia.cpp)
target_link_libraries(test_gum_marsaglia cpproblight zmq)

add_executable(test_gum_marsaglia_replacement test/test_gum_marsaglia_replacement.cpp)
target_link_libraries(test_gum_marsaglia_replacement cpproblight zmq)

add_test(test_gum_1 test_gum)
add_test(test_gum_marsaglia_1 test_gum_marsaglia)
add_test(test_gum_marsaglia_replacement_1 test_gum_marsaglia_replacement)

install (TARGETS cpproblight DESTINATION bin)
install (FILES include/cpproblight.h DESTINATION include)

# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cpproblight.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cpproblight.h
)
