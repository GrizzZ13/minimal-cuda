cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(VectorAdd LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80")

find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_library(kernel STATIC src/kernel.cu)
add_executable(main src/main.cc)
target_link_libraries(main PRIVATE kernel CUDA::cudart)
