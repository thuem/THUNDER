cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
project(thuem)

#Note: if you want to use single precision float number, please don't comment out following line
#otherwise, comment out following line, and change value  "yes" to "no"
#add_definitions(-DUSING_SINGLE_PRECISION=yes)
#set(USING_SINGLE_PRECISION "yes")

OPTION(USING_SINGLE_PRECISION "whether to use single float point number for caculation" OFF) 
set(CMAKE_PREFIX_PATH "external")

if(USING_SINGLE_PRECISION STREQUAL "yes")
    add_definitions(-DUSING_SINGLE_PRECISION)
    set(USING_SINGLE_PRECISION "yes")
else()
    set(USING_SINGLE_PRECISION "no")
endif()

set (COMMON_FLAGS "-Wall -Wno-uninitialized -Wno-deprecated-declarations -Wsign-compare -pthread ${ADDITIONAL_FLAGS}")
set (CMAKE_C_FLAGS "${COMMON_FLAGS} -fno-strict-aliasing")
set (CMAKE_CXX_FLAGS "${COMMON_FLAGS}")


if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif()

#message(FATAL_ERROR "USING SINGLE PRECISION:${USING_SINGLE_PRECISION}")
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

#This line is commented by huabin
#find_package(FFTW REQUIRED)

#########################################################################
#Following lines are added by huabin
if(USING_SINGLE_PRECISION STREQUAL "yes")
    find_path (FFTW_INCLUDES fftw3float/fftw3.h) 
    find_library (FFTWOMP_LIBRARIES NAMES fftw3f_threads fftw3f)
    find_library (FFTW_LIBRARIES NAMES fftw3f)
else()
    find_path (FFTW_INCLUDES fftw3double/fftw3.h)
    find_library (FFTWOMP_LIBRARIES NAMES fftw3_threads fftw3)
    find_library (FFTW_LIBRARIES NAMES fftw3)
endif()
# handle the QUIETLY and REQUIRED arguments and set FFTW_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (FFTW DEFAULT_MSG FFTW_LIBRARIES FFTWOMP_LIBRARIES FFTW_INCLUDES)
find_package_handle_standard_args (FFTW DEFAULT_MSG FFTWOMP_LIBRARIES FFTW_INCLUDES)
mark_as_advanced (FFTW_LIBRARIES FFTWOMP_LIBRARIES FFTW_INCLUDES)
##############################################################################

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
