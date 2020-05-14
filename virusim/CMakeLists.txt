cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

project(virusim)
find_package(OpenMP REQUIRED)
add_executable(virusim virusim.cpp Population.cpp Random.cpp)
target_link_libraries(virusim OpenMP::OpenMP_CXX)
