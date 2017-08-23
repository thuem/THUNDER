cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
project(thuem)

set (COMMON_FLAGS "-Wall -Wno-uninitialized -Wno-deprecated-declarations -Wsign-compare -pthread ${ADDITIONAL_FLAGS}")
set (CMAKE_C_FLAGS "${COMMON_FLAGS} -fno-strict-aliasing")
set (CMAKE_CXX_FLAGS "${COMMON_FLAGS}")

if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif()

option (SANITIZE "Turn on sanitization" OFF)
if (SANITIZE)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    if (NOT APPLE)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lasan")
    endif()
endif()

option (SHENWEI "Build for SHENWEI architecture" OFF)
if (SHENWEI)
    add_definitions(-DSHENWEI=1)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(FFTW REQUIRED)
find_package(GSL REQUIRED)

if (NOT SHENWEI)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-fopenmp" HAS_OPENMP)
    
    if (${HAS_OPENMP})
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
endif()

# includes and linker flags
include_directories(${FFTW_INCLUDES} ${GSL_INCLUDE_DIRS})

link_libraries(${FFTW_LIBRARIES} ${FFTWOMP_LIBRARIES} ${GSL_LIBRARIES} ${GLOG_LIBRARIES} ${CMAKE_DL_LIBS})
#link_libraries(${FFTW_LIBRARIES} ${FFTWOMP_LIBRARIES} ${GSL_LIBRARY} ${GSL_CBLAS_LIBRARY} ${GLOG_LIBRARIES} ${CMAKE_DL_LIBS})
#link_libraries(${FFTW_LIBRARIES} ${FFTWOMP_LIBRARIES} ${GSL_LIBRARY} ${GSL_CBLAS_LIBRARY} ${GLOG_LIBRARIES})

if (ADDITIONAL_LINKER_FLAGS)
    link_libraries(${ADDITIONAL_LINKER_FLAGS})
endif()

set (INCLUDE_BASE ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_BASE}/Functions ${INCLUDE_BASE}/Geometry ${INCLUDE_BASE}/ML ${INCLUDE_BASE}/Image ${INCLUDE_BASE})

set (EXTERNAL_DIR ${PROJECT_SOURCE_DIR}/external)
file (GLOB EXTERNAL_SUBDIR ${EXTERNAL_DIR}/*)
include_directories(${EXTERNAL_SUBDIR})

file(GLOB_RECURSE SOURCES ${EXTERNAL_DIR}/*.c ${EXTERNAL_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(thuem STATIC ${SOURCES})

# Libraries are compiled. Now compile the programs

link_libraries(thuem)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/unittest)
file(GLOB TEST_SOURCES ${PROJECT_SOURCE_DIR}/testsrc/*.cpp)
foreach (TSTSRC ${TEST_SOURCES})
    get_filename_component(TSTNAME ${TSTSRC} NAME_WE)
    add_executable(${TSTNAME} ${TSTSRC})
endforeach()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/app)
file(GLOB APP_SOURCES ${PROJECT_SOURCE_DIR}/appsrc/*.cpp)
foreach (APPSRC ${APP_SOURCES})
    get_filename_component(APPNAME ${APPSRC} NAME_WE)
    add_executable(${APPNAME} ${APPSRC})
endforeach()

file(GLOB_RECURSE CASE_SOURCES ${PROJECT_SOURCE_DIR}/casesrc/*.cpp)
foreach (CASESRC ${CASE_SOURCES})
    get_filename_component(CASENAME ${CASESRC} NAME_WE)
    get_filename_component(CASEDIR ${CASESRC} PATH)
    get_filename_component(CASECATEGORY ${CASEDIR} NAME)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cases/${CASECATEGORY})
    add_executable(${CASENAME} ${CASESRC})
endforeach()
