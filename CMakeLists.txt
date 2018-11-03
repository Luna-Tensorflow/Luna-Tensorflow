cmake_minimum_required(VERSION 3.10)
project(tf_example)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/src/)
include_directories(${CMAKE_SOURCE_DIR}/tensorflow/include/)
link_directories(${CMAKE_SOURCE_DIR}/tensorflow/lib)

add_executable(tf_example src/main.cpp src/Tensor2d.cpp src/Tensor2d.h src/Graph.h)

target_link_libraries(tf_example tensorflow)