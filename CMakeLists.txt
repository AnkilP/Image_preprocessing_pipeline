cmake_minimum_required(VERSION 3.10)
project(image_preprocessing)

set(CMAKE_CXX_STANDARD 17)

add_library(image_preprocessing image_processing.cpp image_processing.hpp)
find_package(OpenCV REQUIRED PATHS /opt/ros/lunar/share/OpenCV-3.3.1-dev)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

