cmake_minimum_required(VERSION 3.5.0)
project(source/GenerateImg)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(include)
include_directories(${OpenCL_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_HOME_DIRECTORY})

add_subdirectory(source/GenerateImg)
add_subdirectory(source/GenerateLine)


