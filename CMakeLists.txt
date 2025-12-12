cmake_minimum_required(VERSION 3.10)
project(lab11-part3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(lab11_part3
    src/main.cpp
    src/matrix.cpp
    src/linear_solvers.cpp
)
