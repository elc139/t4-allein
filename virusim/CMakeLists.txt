cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

project(virusim)
add_executable(virusim virusim.cpp Population.cpp Random.cpp)

