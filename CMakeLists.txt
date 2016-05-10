cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
project(thuem)

set (CMAKE_C_FLAGS "-Wall -fopenmp -march=native -mtune=native")
set (CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++0x")
set (CMAKE_BUILD_TYPE Release)

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

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.c)
add_library(thuem STATIC ${SOURCES})

set (TEST_NAMES TabFunctionTest;ParallelTest;ExperimentTest;ProjectorTest;ExpectationTest;CTFTest;FunctionsTest;DatabaseTest;ImageFunctionsTest;EulerTest;PreprocessTest;ReconstructorTest;SymmetryTest;ParticleTest;TransformationTest)
foreach (TSTEXE ${TEST_NAMES})
	add_executable(${TSTEXE} ${PROJECT_SOURCE_DIR}/testsrc/${TSTEXE}.cpp)
	target_link_libraries(${TSTEXE} thuem)
endforeach()
