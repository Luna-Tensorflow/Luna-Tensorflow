cmake_minimum_required(VERSION 3.12)
project(tf_example)

set(CMAKE_CXX_STANDARD 17)

add_executable(tf_example src/main.cpp)

target_link_libraries(tf_example tensorflow)