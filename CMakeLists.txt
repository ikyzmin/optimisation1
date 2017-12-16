cmake_minimum_required(VERSION 3.9)
project(optimisation1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-lcblas -latlas -lm -fopenmp -lm -Werror -Wfatal-errors -Wall -O2")

add_executable(optimisation1 main.cpp)