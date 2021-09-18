cmake_minimum_required(VERSION 3.20)
project(LBMplayground)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")

find_package(OpenMP)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)
#file(GLOB_RECURSE SOURCE_FILES src/*.cpp)

include_directories(include)

add_executable(LBMplayground src/main.cpp)

target_link_libraries(LBMplayground PRIVATE glfw GLEW::GLEW OpenMP::OpenMP_CXX glm::glm)


add_executable(LBM_ref src/lbm.cpp)
target_link_libraries(LBM_ref PRIVATE glfw GLEW::GLEW OpenMP::OpenMP_CXX glm::glm)
