cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
project(thuem)

set (CMAKE_C_FLAGS "-Wall -fopenmp -march=native -mtune=native")
set (CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++0x")
set (CMAKE_BUILD_TYPE Release)

option (SANITIZE "Turn on sanitization" OFF)
if (SANITIZE)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(FFTW REQUIRED)
find_package(GSL REQUIRED)
find_package(MPI REQUIRED)
find_package(Glog REQUIRED)
include_directories(${FFTW_INCLUDES} ${GSL_INCLUDE_DIRS} ${MPI_INCLUDE_PATH} ${GLOG_INCLUDE_DIR})
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

link_libraries(${FFTW_LIBRARIES} ${GSL_LIBRARIES} ${MPI_LIBRARIES} ${GLOG_LIBRARIES} ${CMAKE_DL_LIBS})

set (INCLUDE_BASE ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_BASE}/Error ${INCLUDE_BASE}/Functions ${INCLUDE_BASE}/Geometry ${INCLUDE_BASE}/ML ${INCLUDE_BASE}/Image ${INCLUDE_BASE})
include_directories(${PROJECT_SOURCE_DIR}/external/Eigen3)

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.c ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(thuem STATIC ${SOURCES})


# Libraries are compiled. Now compile the programs

link_libraries(thuem)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/unittest)
file(GLOB TEST_SOURCES ${PROJECT_SOURCE_DIR}/testsrc/*.cpp)
foreach (TSTSRC ${TEST_SOURCES})
    get_filename_component(TSTNAME ${TSTSRC} NAME_WE)
    add_executable(${TSTNAME} ${TSTSRC})
endforeach()

file(GLOB_RECURSE CASE_SOURCES ${PROJECT_SOURCE_DIR}/casesrc/*.cpp)
foreach (CASESRC ${CASE_SOURCES})
    get_filename_component(CASENAME ${CASESRC} NAME_WE)
    get_filename_component(CASEDIR ${CASESRC} PATH)
    get_filename_component(CASECATEGORY ${CASEDIR} NAME)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cases/${CASECATEGORY})
    add_executable(${CASENAME} ${CASESRC})
endforeach()
