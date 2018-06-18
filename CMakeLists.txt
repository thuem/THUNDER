# CMAKE REQUIREMENT

cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR) 

set(CMAKE_C_FLAGS_DEBUG "" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "")
set(CMAKE_C_FLAGS_RELEASE "" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "")

project(THUNDER)

# THUNDER VERSION

set(THUNDER_VERSION_MAJOR 1)
set(THUNDER_VERSION_MINOR 3)
set(THUNDER_VERSION_ADDIT 3)

message(STATUS "Welcome to THUNDER ${THUNDER_VERSION_MAJOR}.${THUNDER_VERSION_MINOR}.${THUNDER_VERSION_ADDIT}!")

option(SINGLE_PRECISION "Whether to use single float precision in caculation?" ON)

# WHETHER SINGLE PRECISION OR BOTH PRECISION

if("${SINGLE_PRECISION}")
    message(STATUS "Build THUNDER in single-float precision.")
else("${SINGLE_PRECISION}")
    message(STATUS "Build THUNDER in double-float precision.")
endif("${SINGLE_PRECISION}")

# WHETHER GPU VERSION OR CPU VERSION

option(GPU_VERSION "Whether to use gpu version in caculation?" ON)

if("${GPU_VERSION}")
    message(STATUS "Build THUNDER in GPU version.")
else("${GPU_VERSION}")
    message(STATUS "Build THUNDER in GPU version.")
endif("${GPU_VERSION}")

# COMMON FLAGS

set(COMMON_FLAGS "${COMMON_FLAGS} -Wall -Wno-uninitialized -Wno-deprecated-declarations -Wsign-compare -pthread -fopenmp ${ADDITIONAL_FLAGS}")

# BUILD TYPES

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "release")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

if("${CMAKE_BUILD_TYPE}" STREQUAL "release")
    set(COMMON_FLAGS "${COMMON_FLAGS} -O2")
    message(STATUS "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}, compile THUNDER with -O2 flag.")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "debug")
    set(COMMON_FLAGS "${COMMON_FLAGS} -g -O0")
    message(STATUS
            "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}, compile THUNDER with -g -O0 flag.")
else("${CMAKE_BUILD_TYPE}" STREQUAL "release")
    message(FATAL_ERROR
            "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE} is not valid."
            "Valid option is release or debug.")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "release")

# SIMD

option(ENABLE_SIMD "Whether to use SIMD to accelerate?" ON)
option(ENABLE_AVX512 "Whether to use AVX512 to accelerate?" OFF)
option(ENABLE_AVX256 "Whether to use AVX256 to accelerate?" ON)

if("${ENABLE_SIMD}")
    if(APPLE)
        message(WARNING "THUNDER does not suppot SIMD on Apple.")
        set(ENABLE_SIMD OFF)
        set(CMAKE_C_FLAGS "${COMMON_FLAGS}")
        set(CMAKE_CXX_FLAGS "${COMMON_FLAGS}")
    else(APPLE)
        set(CMAKE_C_FLAGS "${COMMON_FLAGS} -mavx512f -mavx512cd")
        set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} -mavx512f -mavx512cd")
        try_compile(AVX512_SUPPORT
                    ${CMAKE_BINARY_DIR}
                    "${CMAKE_SOURCE_DIR}/cmake/SIMD/AVX512.c")
        if(AVX512_SUPPORT AND ENABLE_AVX512)
            message(STATUS "Build THUNDER using AVX512.")
            set(ENABLE_SIMD_512 ON)
        else(AVX512_SUPPORT AND ENABLE_AVX512)
            if(ENABLE_AVX512)
                message(WARNING "Compiler does not support AVX512.")
            endif(ENABLE_AVX512)
            set(CMAKE_C_FLAGS "${COMMON_FLAGS} -mavx")
            set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} -mavx")
            try_compile(AVX256_SUPPORT
                        ${CMAKE_BINARY_DIR}
                        "${CMAKE_SOURCE_DIR}/cmake/SIMD/AVX256.c")
            if(AVX256_SUPPORT AND ENABLE_AVX256)
                message(STATUS "Build THUNDER using AVX256.")
                set(ENABLE_SIMD_256 ON)
            else(AVX256_SUPPORT AND ENABLE_AVX256)
                if(ENABLE_AVX256)
                    message(WARNING "Compiler does not support AVX256.")
                endif(ENABLE_AVX256)
                set(ENABLE_SIMD OFF)
                set(CMAKE_C_FLAGS "${COMMON_FLAGS}")
                set(CMAKE_CXX_FLAGS "${COMMON_FLAGS}")
            endif(AVX256_SUPPORT AND ENABLE_AVX256)
        endif(AVX512_SUPPORT AND ENABLE_AVX512)
    endif(APPLE)
else("${ENABLE_SIMD}")
    message(STATUS "Build THUNDER without using SIMD.")
    set(CMAKE_C_FLAGS "${COMMON_FLAGS}")
    set(CMAKE_CXX_FLAGS "${COMMON_FLAGS}")
endif("${ENABLE_SIMD}")

# OUTPUT FLAGS

message(STATUS "CMAKE_C_FLAGS : ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")

# MPI

find_package(MPI REQUIRED)

#include_directories("${MPI_INCLUDE_PATH}") # Old Version Support

message(STATUS "MPI_COMPILER : ${MPI_COMPILER}") # Old Version Support
message(STATUS "MPI_INCLUDE_PATH : ${MPI_INCLUDE_PATH}") # Old Version Support
message(STATUS "MPI_LIBRARIES : ${MPI_LIBRARIES}") # Old Version Support
message(STATUS "MPI_CXX_INCLUDE_PATH : ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "MPI_CXX_COMPILER : ${MPI_CXX_COMPILER}")
message(STATUS "MPI_CXX_LIBRARIES : ${MPI_CXX_LIBRARIES}")

set(CMAKE_C_COMPILER ${MPI_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

# CONFIG

configure_file("${PROJECT_SOURCE_DIR}/THUNDERConfig.h.in"
               "${PROJECT_BINARY_DIR}/THUNDERConfig.h")

install(FILES "${PROJECT_BINARY_DIR}/THUNDERConfig.h" DESTINATION include)

include_directories(${PROJECT_BINARY_DIR})

# CMAKE MODULE FILE LOCATION

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

message(STATUS "Getting ready for building FFTW-3")

include(${PROJECT_SOURCE_DIR}/cmake/BuildFFTW.cmake)

# ADD HEADER OF FFTW

include_directories(BEFORE ${PROJECT_BINARY_DIR}/external/fftw/include)

message(STATUS "Getting ready for building GSL")

# BUILD GSL

include(${PROJECT_SOURCE_DIR}/cmake/BuildGSL.cmake)

# ADD HEADER OF GSL

include_directories(BEFORE ${PROJECT_BINARY_DIR}/external/gsl/include)

# ADD HEADER OF EIGEN3

include_directories(${PROJECT_SOURCE_DIR}/external/Eigen3)

# ADD HEADER OF JSONCPP

include_directories(${PROJECT_SOURCE_DIR}/external/jsoncpp)

# COMPILE LIBRARY OF JSONCPP

message(STATUS "Getting ready for building jsoncpp")

add_subdirectory(${PROJECT_SOURCE_DIR}/external/jsoncpp)

# ADD HEADER OF BOOST

include_directories(${PROJECT_SOURCE_DIR}/external/boost)

# ADD HEADER OF EASYLOGGING

include_directories(${PROJECT_SOURCE_DIR}/external/easylogging)

set(INCLUDE_BASE ${PROJECT_SOURCE_DIR}/include)
set(CUR_INCLUDE ${INCLUDE_BASE})

list(APPEND CUR_INCLUDE ${INCLUDE_BASE}/Functions)
list(APPEND CUR_INCLUDE ${INCLUDE_BASE}/Geometry)
list(APPEND CUR_INCLUDE ${INCLUDE_BASE}/Image)

# CUDA & NCCL FOR GPU VERSION

if("${GPU_VERSION}")

    # CUDA

    find_package(CUDA 8.0 REQUIRED)

    message(STATUS "CUDA_VERSION :" ${CUDA_VERSION})
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR : " ${CUDA_TOOLKIT_ROOT_DIR})
    message(STATUS "CUDA_INCLUDE_DIRS : " ${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_LIBRARIES : " ${CUDA_LIBRARIES})
    message(STATUS "CUDA_CUBLAS_LIBRARIES : " ${CUDA_CUBLAS_LIBRARIES})
    message(STATUS "CUDA_curand_LIBRARY : " ${CUDA_curand_LIBRARY})
    message(STATUS "CUDA_CUFFT_LIBRARIES : " ${CUDA_CUFFT_LIBRARIES})

    # LINK THE CUDA HEADER

    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)

    # THE GPU ARCHITETURE SPECIFIC BINARY

    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_35,code=sm_35")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=sm_50")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_53,code=sm_53")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60")
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

    # LINK THE CUDA LIBRARIES

    list(APPEND EXTERNAL_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    list(APPEND EXTERNAL_LIBS ${CUDA_CUFFT_LIBRARIES})
    list(APPEND EXTERNAL_LIBS ${CUDA_curand_LIBRARY})

    # NCCL

    find_package(NCCL REQUIRED)

    message(STATUS "NCCL_INCLUDES : " ${NCCL_INCLUDES})
    message(STATUS "NCCL_LIBRARIES : " ${NCCL_LIBRARIES})

    # LINK THE NCCL HEADER

    include_directories(SYSTEM ${NCCL_INCLUDES})

    # LINK THE NCLL LIBRARIES

    list(APPEND EXTERNAL_LIBS ${NCCL_LIBRARIES})

 endif("${GPU_VERSION}")

# Build routine
add_subdirectory(gpu)

# Make Sure FFTW and GSL is Built Before Linking

add_dependencies(cuthuem FFTW)
add_dependencies(cuthuem GSL)

endif("${GPU_VERSION}")

install(DIRECTORY ${PROJECT_BINARY_DIR}/external/fftw/lib/ DESTINATION lib FILES_MATCHING PATTERN "*")
install(DIRECTORY ${PROJECT_BINARY_DIR}/external/gsl/lib/ DESTINATION lib FILES_MATCHING PATTERN "*")

# Add Header of THUNDER core

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/include/Functions
                    ${PROJECT_SOURCE_DIR}/include/Geometry
                    ${PROJECT_SOURCE_DIR}/include/Image)

# Compile Binary of THUNDER core

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(core STATIC ${SOURCES})

# Make Sure FFTW and GSL is Built Before Linking

add_dependencies(core FFTW)
add_dependencies(core GSL)

# Link Libraries

link_libraries(core ${FFTW_LIBRARIES} ${GSL_LIBRARIES} jsoncpp)

TARGET_include_directories(core PUBLIC ${CUR_INCLUDE} ${EXTERNAL_INCLUDE} ${INTERFACE_INCLUDE})

if("${GPU_VERSION}")
TARGET_LINK_LIBRARIES(core cuthuem ${EXTERNAL_LIBS} ${CMAKE_DL_LIBS})
endif("${GPU_VERSION}")

#ADD_SUBDIRECTORY(testsrc)

# Compile Executables

file(GLOB BIN_SOURCES ${PROJECT_SOURCE_DIR}/appsrc/*.cpp)

foreach(BINSRC ${BIN_SOURCES})
    get_filename_component(BINNAME ${BINSRC} NAME_WE) # file name without directories or longest extension
    add_executable(${BINNAME} ${BINSRC})
    install(TARGETS ${BINNAME} RUNTIME DESTINATION bin)
endforeach()

# Copy Scripts

install(FILES "${PROJECT_SOURCE_DIR}/script/STAR_2_THU.py" DESTINATION script)
install(FILES "${PROJECT_SOURCE_DIR}/script/demo.json" DESTINATION script)
install(FILES "${PROJECT_SOURCE_DIR}/script/demo_2D.json" DESTINATION script)
install(FILES "${PROJECT_SOURCE_DIR}/script/demo_3D.json" DESTINATION script)

# Copy Manual

install(FILES "${PROJECT_SOURCE_DIR}/manual/THUNDER_v${THUNDER_VERSION_MAJOR}_${THUNDER_VERSION_MINOR}_${THUNDER_VERSION_ADDIT}_User_Guide.pdf" DESTINATION manual)
