cmake_minimum_required(VERSION 3.17)
project(Perceptron)

set(CMAKE_CXX_STANDARD 17)

add_executable(Perceptron main.cpp neuron.cpp neuron.h)