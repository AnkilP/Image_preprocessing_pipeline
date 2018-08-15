cmake_minimum_required(VERSION 3.12)
project(Image_preprocessing_pipeline)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
link_directories(.)

add_executable(Image_preprocessing_pipeline adaptive_manifold.cpp adaptive_manifold.hpp)

target_link_libraries(Image_preprocessing_pipeline gtest gtest_main)

