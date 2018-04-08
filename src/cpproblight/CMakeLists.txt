project(libcpproblight VERSION 0.1 LANGUAGES CXX)

set (VERSION "0.1.3")

add_library(cpproblight SHARED src/cpproblight.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_local_sample test/test_local_sample.cpp)
target_link_libraries(test_local_sample cpproblight zmq)

add_executable(test_set_defaults_and_addresses test/test_set_defaults_and_addresses.cpp)
target_link_libraries(test_set_defaults_and_addresses cpproblight zmq)

add_executable(test_gum test/test_gum.cpp)
target_link_libraries(test_gum cpproblight zmq)

add_executable(test_gum_marsaglia test/test_gum_marsaglia.cpp)
target_link_libraries(test_gum_marsaglia cpproblight zmq)

add_executable(test_gum_marsaglia_replacement test/test_gum_marsaglia_replacement.cpp)
target_link_libraries(test_gum_marsaglia_replacement cpproblight zmq)

add_executable(test_hmm test/test_hmm.cpp)
target_link_libraries(test_hmm cpproblight zmq)

add_executable(test_branching test/test_branching.cpp)
target_link_libraries(test_branching cpproblight zmq)

add_test(test_local_sample_1 test_local_sample)
add_test(test_set_defaults_and_addresses_1 test_set_defaults_and_addresses)
add_test(test_gum_1 test_gum)
add_test(test_gum_marsaglia_1 test_gum_marsaglia)
add_test(test_gum_marsaglia_replacement_1 test_gum_marsaglia_replacement)
add_test(test_hmm_1 test_hmm)
add_test(test_branching_1 test_branching)

install (TARGETS cpproblight DESTINATION lib)
install (FILES include/cpproblight.h include/pplprotocol.h DESTINATION include)

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
