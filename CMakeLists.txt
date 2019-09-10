# @file
# @author  
# @author Fande Yu 
# @author Mingxu
# @version 1.4.14.090628
# @copyright THUNDER Non-Commercial Software License Agreement
#
# ChangeLog
# AUTHOR      | TIME       | VERSION       | DESCRIPTION
# ------      | ----       | -------       | -----------
# Fande Yu    | 2019/03/25 | 1.4.11.090325 | separate build version in CPU & GPU
# Mingxu Hu   | 2019/05/07 | 1.4.12.090507 | prepare of 1.4.12 release
# Mingxu Hu   | 2019/05/31 | 1.4.12.090531 | -Wno-sign-compare
# Fande Yu    | 2019/06/26 | 1.4.12.090626 | fix compilation problem with cmake below version3.7
# Mingxu Hu   | 2019/06/28 | 1.4.14.090628 | v1.4.14
# Mingxu Hu   | 2019/06/29 | 1.4.14.090629 | googletest added

# CMAKE REQUIREMENT

cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR) 

execute_process(
    COMMAND git rev-parse HEAD 
    OUTPUT_VARIABLE COMMIT_VERSION 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(COMMIT_VERSION_QUOTE "\"${COMMIT_VERSION}\"")

set(CMAKE_C_FLAGS_DEBUG "" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "")
set(CMAKE_C_FLAGS_RELEASE "" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "")

project(THUNDER)

# THUNDER VERSION

set(THUNDER_VERSION_MAJOR 1)
set(THUNDER_VERSION_MINOR 4)
set(THUNDER_VERSION_ADDIT 14)

message(STATUS "Welcome to THUNDER ${THUNDER_VERSION_MAJOR}.${THUNDER_VERSION_MINOR}.${THUNDER_VERSION_ADDIT}!")

message(STATUS "Git commit version:" ${COMMIT_VERSION})

option(SINGLE_PRECISION "Whether to use single float precision in caculation?" ON)

# WHETHER SINGLE PRECISION OR BOTH PRECISION

if("${SINGLE_PRECISION}")
    message(STATUS "Build THUNDER in single-float precision.")
else("${SINGLE_PRECISION}")
    message(STATUS "Build THUNDER in double-float precision.")
endif("${SINGLE_PRECISION}")

# DECIDE THE VERSION AND TOOLS

option(GPU_VERSION "Whether to use gpu version in caculation?" ON)
option(CPU_VERSION "Whether to use cpu version in caculation?" ON)
option(TOOLS_BUILD "Whether to build tools?" ON)
option(UNITTEST_BUILD "Whether to build unittests?" OFF)

if("${GPU_VERSION}" AND "${CPU_VERSION}")

    message(STATUS "Try to build THUNDER in CPU & GPU version.")

else("${GPU_VERSION}" AND "${CPU_VERSION}")

    if("${GPU_VERSION}")
        message(STATUS "Try to build THUNDER in GPU version.")
    else("${GPU_VERSION}")
        message(STATUS "Try to build THUNDER in CPU version.")
    endif("${GPU_VERSION}")

endif("${GPU_VERSION}" AND "${CPU_VERSION}")
# COMMON FLAGS

# set(COMMON_FLAGS "-std=c++11 ${COMMON_FLAGS} -Wall -Wno-uninitialized -Wno-deprecated-declarations -Wno-sign-compare -pthread -fopenmp ${ADDITIONAL_FLAGS}")
set(COMMON_FLAGS "${COMMON_FLAGS} -Wall -Wno-uninitialized -Wno-deprecated-declarations -Wno-sign-compare -pthread -fopenmp ${ADDITIONAL_FLAGS}")
# set(COMMON_FLAGS "${COMMON_FLAGS} -Wall -Wno-uninitialized -Wno-deprecated-declarations -pthread -fopenmp ${ADDITIONAL_FLAGS}")

if("${UNITTEST_BUILD}")
    set(COMMON_FLAGS "-std=c++11 ${COMMON_FLAGS}")
endif("${UNITTEST_BUILD}")

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

# BUILD BOOST

include(${PROJECT_SOURCE_DIR}/cmake/BuildBoost.cmake)

# ADD HEADER OF BOOST

include_directories(BEFORE ${PROJECT_BINARY_DIR}/external/boost)

if("${UNITTEST_BUILD}")

    message(STATUS "Getting ready for building googletest")

    # BUILD GOOGLETEST

    include(${PROJECT_SOURCE_DIR}/cmake/BuildGTest.cmake)

    # ADD HEADER OF GOOGLETEST

    include_directories(BEFORE ${PROJECT_BINARY_DIR}/external/googletest/include)

endif("${UNITTEST_BUILD}")

# ADD HEADER OF EIGEN3

include_directories(${PROJECT_SOURCE_DIR}/external/Eigen3)

# ADD HEADER OF JSONCPP

include_directories(${PROJECT_SOURCE_DIR}/external/jsoncpp)

# COMPILE LIBRARY OF JSONCPP

message(STATUS "Getting ready for building jsoncpp")

add_subdirectory(${PROJECT_SOURCE_DIR}/external/jsoncpp)

# ADD HEADER OF EASYLOGGING

include_directories(${PROJECT_SOURCE_DIR}/external/easylogging)

set(INCLUDE_BASE ${PROJECT_SOURCE_DIR}/include)
set(CUR_INCLUDE ${INCLUDE_BASE})

list(APPEND CUR_INCLUDE ${INCLUDE_BASE}/Functions)
list(APPEND CUR_INCLUDE ${INCLUDE_BASE}/Geometry)
list(APPEND CUR_INCLUDE ${INCLUDE_BASE}/Image)

# CUDA & NCCL FOR GPU VERSION

find_package(CUDA 8.0)
if(NOT ${CUDA_FOUND})
    if("${GPU_VERSION}")
        message(STATUS "WARNING: CUDA is not found, skip the compilation in GPU version.")
    endif("${GPU_VERSION}")
    set(GPU_VERSION off)
endif(NOT ${CUDA_FOUND})

find_package(NCCL)
if(NOT ${NCCL_FOUND})
    if("${GPU_VERSION}")
        message(STATUS "WARNING: NCCL is not found, skip the compilation in GPU version.")
    endif("${GPU_VERSION}")
    set(GPU_VERSION off)
endif(NOT ${NCCL_FOUND})

# Add Header of THUNDER core_CPU and core_GPU

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/include/Functions
                    ${PROJECT_SOURCE_DIR}/include/Geometry
                    ${PROJECT_SOURCE_DIR}/include/Image)
# CONFIG

if("${CPU_VERSION}")

    set(CONFIG_CPU_DIR "${PROJECT_BINARY_DIR}/Config_CPU")
    
    # If CPU_VERSION and GPU_VERSION is set ON in the meantime
    if("${GPU_VERSION}")
        set(GPU_VERSION off)

        configure_file("${PROJECT_SOURCE_DIR}/THUNDERConfig.h.in"
                   "${CONFIG_CPU_DIR}/THUNDERConfig.h")
    
        #install(FILES "${PROJECT_BINARY_DIR}/THUNDERConfig.h" DESTINATION include)
    
        set(GPU_VERSION on)
    
    else("${GPU_VERSION}")

        configure_file("${PROJECT_SOURCE_DIR}/THUNDERConfig.h.in"
                   "${CONFIG_CPU_DIR}/THUNDERConfig.h")
    
        #install(FILES "${PROJECT_BINARY_DIR}/THUNDERConfig.h" DESTINATION include)
        
    endif("${GPU_VERSION}")    

    # Compile Binary of THUNDER core_CPU
    
    file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
    
    add_library(core_CPU STATIC ${SOURCES})
    
    # Make Sure FFTW and GSL is Built Before Linking
    
    add_dependencies(core_CPU FFTW)
    add_dependencies(core_CPU GSL)
    add_dependencies(core_CPU BOOST)
    
    # Link Libraries
    
    target_link_libraries(core_CPU ${FFTW_LIBRARIES} ${GSL_LIBRARIES} jsoncpp)
    
    target_include_directories(core_CPU PUBLIC ${CUR_INCLUDE} ${CONFIG_CPU_DIR})

endif("${CPU_VERSION}")

if("${GPU_VERSION}")

    set(CONFIG_GPU_DIR ${PROJECT_BINARY_DIR}/Config_GPU)
    
    configure_file("${PROJECT_SOURCE_DIR}/THUNDERConfig.h.in"
               "${CONFIG_GPU_DIR}/THUNDERConfig.h")

    #install(FILES "${PROJECT_BINARY_DIR}/THUNDERConfig.h" DESTINATION include)
    
    # Compile Binary of THUNDER core_GPU
    
    file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
    
    add_library(core_GPU STATIC ${SOURCES})
    
    # Make Sure FFTW and GSL is Built Before Linking
    
    add_dependencies(core_GPU FFTW)
    add_dependencies(core_GPU GSL)
    add_dependencies(core_GPU BOOST)
    
    # Link Libraries
    
    target_link_libraries(core_GPU ${FFTW_LIBRARIES} ${GSL_LIBRARIES} jsoncpp)
    
    target_include_directories(core_GPU PUBLIC ${CUR_INCLUDE} ${CONFIG_GPU_DIR})
    
endif("${GPU_VERSION}")

if("${GPU_VERSION}")

    # CUDA

    message(STATUS "CUDA_VERSION : " ${CUDA_VERSION})
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR : " ${CUDA_TOOLKIT_ROOT_DIR})
    message(STATUS "CUDA_INCLUDE_DIRS : " ${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_LIBRARIES : " ${CUDA_LIBRARIES})
    message(STATUS "CUDA_CUBLAS_LIBRARIES : " ${CUDA_CUBLAS_LIBRARIES})
    message(STATUS "CUDA_curand_LIBRARY : " ${CUDA_curand_LIBRARY})
    message(STATUS "CUDA_CUFFT_LIBRARIES : " ${CUDA_CUFFT_LIBRARIES})

    # LINK THE CUDA HEADER

    target_include_directories(core_GPU PUBLIC SYSTEM ${CUDA_INCLUDE_DIRS})
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)

    # THE GPU ARCHITETURE SPECIFIC BINARY

    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_35,code=sm_35")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=sm_50")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_53,code=sm_53")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60")
    if(("${CUDA_VERSION}" STRGREATER "9.0") OR ("${CUDA_VERSION}" STREQUAL "9.0"))
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_70,code=sm_70")
    endif(("${CUDA_VERSION}" STRGREATER "9.0") OR ("${CUDA_VERSION}" STREQUAL "9.0"))
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
    list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")

    message(STATUS "CUDA_NVCC_FLAGS : " ${CUDA_NVCC_FLAGS})

    # LINK THE CUDA LIBRARIES

    list(APPEND EXTERNAL_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    list(APPEND EXTERNAL_LIBS ${CUDA_CUFFT_LIBRARIES})
    list(APPEND EXTERNAL_LIBS ${CUDA_curand_LIBRARY})
    
    if(("${CMAKE_MINOR_VERSION}" STRLESS "7") OR ("${CMAKE_MINOR_VERSION}" STREQUAL "7"))
        find_library(CUDA_cudadevrt_LIBRARY libcudadevrt.a PATHS ${CUDA_TOOLKIT_ROOT_DIR})
        list(APPEND EXTERNAL_LIBS ${CUDA_cudadevrt_LIBRARY})
        #message(STATUS "CUDA_cudadevrt_LIBRARY:${CUDA_cudadevrt_LIBRARY}")
    endif(("${CMAKE_MINOR_VERSION}" STRLESS "7") OR ("${CMAKE_MINOR_VERSION}" STREQUAL "7"))

    # NCCL

    message(STATUS "NCCL_INCLUDES : " ${NCCL_INCLUDES})
    message(STATUS "NCCL_LIBRARIES : " ${NCCL_LIBRARIES})

    # LINK THE NCCL HEADER

    target_include_directories(core_GPU PUBLIC SYSTEM ${NCCL_INCLUDES})

    # LINK THE NCLL LIBRARIES

    list(APPEND EXTERNAL_LIBS ${NCCL_LIBRARIES})

    # BUILD GPU SUBROUNTINE

    add_subdirectory(gpu)

    # MAKE SURE FFTW AND GSL IS BUILT BEFORE LINKING

    add_dependencies(cuthuem FFTW)
    add_dependencies(cuthuem GSL)

endif("${GPU_VERSION}")

install(DIRECTORY ${PROJECT_BINARY_DIR}/external/fftw/lib/ DESTINATION lib FILES_MATCHING PATTERN "*")
install(DIRECTORY ${PROJECT_BINARY_DIR}/external/gsl/lib/ DESTINATION lib FILES_MATCHING PATTERN "*")

# Compile Executables

file(GLOB BIN_SOURCES ${PROJECT_SOURCE_DIR}/appsrc/*.cpp)

if("${CPU_VERSION}")
    foreach(BINSRC ${BIN_SOURCES})
    
        get_filename_component(BINNAME ${BINSRC} NAME_WE) # file name without directories or longest extension
        if("${BINNAME}" STREQUAL "thunder")
            set(BINNAME "${BINNAME}_cpu")
        endif("${BINNAME}" STREQUAL "thunder")
        
        if(("${BINNAME}" STREQUAL "thunder_cpu") OR ("${TOOLS_BUILD}"))
            add_executable(${BINNAME} ${BINSRC})
            target_link_libraries(${BINNAME} core_CPU)
            target_include_directories(${BINNAME} PUBLIC ${CONFIG_CPU_DIR} ${CUR_INCLUDE})
            install(TARGETS ${BINNAME} RUNTIME DESTINATION bin)
        endif(("${BINNAME}" STREQUAL "thunder_cpu") OR ("${TOOLS_BUILD}"))
    
    endforeach()
    
    # BUILD TOOL ONLY ONCE
    if("${TOOLS_BUILD}")
        set(TOOLS_BUILD OFF)
    endif("${TOOLS_BUILD}")

    # BUILD UNIT TEST
    
    if(${UNITTEST_BUILD})

        enable_testing()

        # find_package(GTest REQUIRED)

        # message(STATUS "GTEST_INCLUDE_DIRS : " ${GTEST_INCLUDE_DIRS})
        # message(STATUS "GTEST_BOTH_LIBRARIES : " ${GTEST_BOTH_LIBRARIES})
        # message(STATUS "GTEST_LIBRARIES : " ${GTEST_LIBRARIES})
        # message(STATUS "GTEST_MAIN_LIBRARIES : " ${GTEST_MAIN_LIBRARIES})

        # include_directories("${GTEST_PATH}/include")

        # target_link_libraries(core_CPU ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

        file(GLOB UNIT_SOURCES ${PROJECT_SOURCE_DIR}/unitsrc/*.cpp)

        foreach(UNITSRC ${UNIT_SOURCES})

            message(STATUS "UNITSRC : " ${UNITSRC})

            get_filename_component(UNITNAME ${UNITSRC} NAME_WE) # file name without directories or longest extension

            add_executable(${UNITNAME} ${UNITSRC})

            # target_link_libraries(${UNITNAME} core_CPU ${GTEST_BOTH_LIBRARIES})
            # target_link_libraries(${UNITNAME} core_CPU ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
            target_link_libraries(${UNITNAME} core_CPU ${GTEST_LIBRARIES})

            target_include_directories(${UNITNAME} PUBLIC ${CONFIG_CPU_DIR} ${CUR_INCLUDE})

            # add_test(AllTests ${UNITNAME})

            install(TARGETS ${UNITNAME} RUNTIME DESTINATION bin)
        endforeach()
    endif(${UNITTEST_BUILD})

endif("${CPU_VERSION}")

file(GLOB BIN_SOURCES ${PROJECT_SOURCE_DIR}/appsrc/*.cpp)

if("${GPU_VERSION}")
    
    message(STATUS "${GPU_VERSION}") 
    
    target_link_libraries(core_GPU ${FFTW_LIBRARIES} ${GSL_LIBRARIES} jsoncpp)
    
    target_include_directories(core_GPU PUBLIC ${CUR_INCLUDE} ${EXTERNAL_INCLUDE} ${INTERFACE_INCLUDE})
    
    TARGET_LINK_LIBRARIES(core_GPU cuthuem ${EXTERNAL_LIBS} ${CMAKE_DL_LIBS})
    #LINK_LIBRARIES(core cuthuem ${EXTERNAL_LIBS} ${CMAKE_DL_LIBS})
    #TARGET_LINK_LIBRARIES(core ${EXTERNAL_LIBS} ${CMAKE_DL_LIBS})

    foreach(BINSRC ${BIN_SOURCES})

        get_filename_component(BINNAME ${BINSRC} NAME_WE) # file name without directories or longest extension

        if("${BINNAME}" STREQUAL "thunder")
            set(BINNAME "${BINNAME}_gpu")
        endif("${BINNAME}" STREQUAL "thunder")
        
        if(("${BINNAME}" STREQUAL "thunder_gpu") OR ("${TOOLS_BUILD}"))
            add_executable(${BINNAME} ${BINSRC})
            if("${BINNAME}" STREQUAL "thunder_gpu")
                target_link_libraries(${BINNAME} core_GPU)
                target_include_directories(${BINNAME} PUBLIC SYSTEM ${NCCL_INCLUDE} ${CUDA_INCLUDE_DIR})
            endif("${BINNAME}" STREQUAL "thunder_gpu")
            install(TARGETS ${BINNAME} RUNTIME DESTINATION bin)
        endif(("${BINNAME}" STREQUAL "thunder_gpu") OR ("${TOOLS_BUILD}"))

    endforeach()

endif("${GPU_VERSION}")
# Copy Scripts

install(FILES "${PROJECT_SOURCE_DIR}/script/STAR_2_THU.py" DESTINATION script)
install(FILES "${PROJECT_SOURCE_DIR}/script/demo.json" DESTINATION script)
install(FILES "${PROJECT_SOURCE_DIR}/script/demo_2D.json" DESTINATION script)
install(FILES "${PROJECT_SOURCE_DIR}/script/demo_3D.json" DESTINATION script)
