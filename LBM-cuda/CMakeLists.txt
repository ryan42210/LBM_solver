cmake_minimum_required(VERSION 3.20)
project(LBM_cuda LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(include)

add_executable(LBM_cuda src/main.cu)

set_target_properties(LBM_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(LBM_cuda PRIVATE glfw GLEW::GLEW OpenMP::OpenMP_CXX)