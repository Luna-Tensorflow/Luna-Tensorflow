cmake_minimum_required(VERSION 3.2)
include(CheckCXXCompilerFlag)
project(tf_in_c)

message("Make sure to download and extract an appropriate version of Tensorflow into tensorflow/ directory")

set(CMAKE_CXX_STANDARD 14)
include_directories(${CMAKE_SOURCE_DIR}/tensorflow/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/)
link_directories(${CMAKE_SOURCE_DIR}/tensorflow/lib)

set(API_SOURCES
  src/tf/version.cc
  src/tf/version.h
  src/tf/tensor.cc
  src/tf/tensor.h
)

add_executable(example_tf
  src/example_main.cc
  ${API_SOURCES}
)
target_link_libraries(example_tf tensorflow)
