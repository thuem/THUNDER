cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
project(thuem)

set (CMAKE_CXX_FLAGS "-std=c++0x -Wall -Wextra -fopenmp -march=native -mtune=native")
set (CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(FFTW REQUIRED)
find_package(GSL REQUIRED)
include_directories(${FFTW_INCLUDES} ${GSL_INCLUDE_DIRS})

set (INCLUDE_BASE ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_BASE} ${INCLUDE_BASE}/Error ${INCLUDE_BASE}/Functions ${INCLUDE_BASE}/Geometry ${INCLUDE_BASE}/ML ${INCLUDE_BASE}/Image)
include_directories(${PROJECT_SOURCE_DIR}/external/Eigen3)

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(libthuem STATIC ${SOURCES})
