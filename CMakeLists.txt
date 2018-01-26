cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR) 

set(CMAKE_C_FLAGS_DEBUG "" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "")
set(CMAKE_C_FLAGS_RELEASE "" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "")

project(THUNDER)

set(THUNDER_VERSION_MAJOR 1)
set(THUNDER_VERSION_MINOR 3)
set(THUNDER_VERSION_ADDIT 1)

message(STATUS "Welcome to THUNDER ${THUNDER_VERSION_MAJOR}.${THUNDER_VERSION_MINOR}.${THUNDER_VERSION_ADDIT}!")

option(SINGLE_PRECISION "Whether to use single float precision in caculation?" ON)

if ("${SINGLE_PRECISION}")
    message(STATUS "Build THUNDER in single-float precision.")
else("${SINGLE_PRECISION}")
    message(STATUS "Build THUNDER in double-float precision.")
endif("${SINGLE_PRECISION}")

set(COMMON_FLAGS "${COMMON_FLAGS} -Wall -Wno-uninitialized -Wno-deprecated-declarations -Wsign-compare -pthread -fopenmp ${ADDITIONAL_FLAGS}")

# Build Type

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
            if(NOT AVX512_SUPPORT)
                message(WARNING "Compiler does not support AVX512.")
            endif(NOT AVX512_SUPPORT)
            set(CMAKE_C_FLAGS "${COMMON_FLAGS} -mavx")
            set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} -mavx")
            try_compile(AVX256_SUPPORT
                        ${CMAKE_BINARY_DIR}
                        "${CMAKE_SOURCE_DIR}/cmake/SIMD/AVX256.c")
            if(AVX256_SUPPORT AND ENABLE_AVX256)
                message(STATUS "Build THUNDER using AVX256.")
                set(ENABLE_SIMD_256 ON)
            else(AVX256_SUPPORT AND ENABLE_AVX256)
                if(NOT AVX256_SUPPORT)
                    message(WARNING "Compiler does not support AVX256.")
                endif(NOT AVX256_SUPPORT)
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

# Output Flags
message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# MPI

find_package(MPI REQUIRED)

include_directories("${MPI_INCLUDE_PATH}") # Old Version Support

message(STATUS "MPI_COMPILER : ${MPI_COMPILER}") # Old Version Support
message(STATUS "MPI_INCLUDE_PATH : ${MPI_INCLUDE_PATH}") # Old Version Support
message(STATUS "MPI_LIBRARIES : ${MPI_LIBRARIES}") # Old Version Support
message(STATUS "MPI_CXX_INCLUDE_PATH : ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "MPI_CXX_COMPILER : ${MPI_CXX_COMPILER}")
message(STATUS "MPI_CXX_LIBRARIES : ${MPI_CXX_LIBRARIES}")

set(CMAKE_C_COMPILER ${MPI_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

# Config

configure_file("${PROJECT_SOURCE_DIR}/THUNDERConfig.h.in"
               "${PROJECT_BINARY_DIR}/THUNDERConfig.h")

install(FILES "${PROJECT_BINARY_DIR}/THUNDERConfig.h" DESTINATION include)

include_directories(${PROJECT_BINARY_DIR})

message(STATUS "Getting ready for building FFTW-3")

include(${PROJECT_SOURCE_DIR}/cmake/BuildFFTW.cmake)

# Add Header of FFTW

include_directories(BEFORE ${PROJECT_BINARY_DIR}/external/fftw/include)

message(STATUS "Getting ready for building GSL")

# Build GSL

include(${PROJECT_SOURCE_DIR}/cmake/BuildGSL.cmake)

# Add Header of GSL

include_directories(BEFORE ${PROJECT_BINARY_DIR}/external/gsl/include)

# Add Header of Eigen3

include_directories(${PROJECT_SOURCE_DIR}/external/Eigen3)

# Add Header of jsoncpp

include_directories(${PROJECT_SOURCE_DIR}/external/jsoncpp)

# Compile Libary of jsoncpp

message(STATUS "Getting ready for building jsoncpp")

add_subdirectory(${PROJECT_SOURCE_DIR}/external/jsoncpp)

# Add Header of boost

include_directories(${PROJECT_SOURCE_DIR}/external/boost)

# Add Header of easylogging

include_directories(${PROJECT_SOURCE_DIR}/external/easylogging)

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

install(DIRECTORY ${PROJECT_BINARY_DIR}/external/fftw/lib/ DESTINATION lib FILES_MATCHING PATTERN "*")
install(DIRECTORY ${PROJECT_BINARY_DIR}/external/gsl/lib/ DESTINATION lib FILES_MATCHING PATTERN "*")

# Link Libraries

link_libraries(core ${FFTW_LIBRARIES} ${GSL_LIBRARIES} jsoncpp)

# Compile Executables

file(GLOB BIN_SOURCES ${PROJECT_SOURCE_DIR}/appsrc/*.cpp)

foreach(BINSRC ${BIN_SOURCES})
    get_filename_component(BINNAME ${BINSRC} NAME_WE) # file name without directories or longest extension
    add_executable(${BINNAME} ${BINSRC})
    install(TARGETS ${BINNAME} RUNTIME DESTINATION bin)
endforeach()

# Copy Scripts

install(FILES "${PROJECT_SOURCE_DIR}/script/STAR_2_THU.py" DESTINATION script)
install(FILES "${PROJECT_SOURCE_DIR}/script/STAR_2_THU_NO_GROUP.py" DESTINATION script)
install(FILES "${PROJECT_SOURCE_DIR}/script/demo.json" DESTINATION script)
install(FILES "${PROJECT_SOURCE_DIR}/script/demo_2D.json" DESTINATION script)
install(FILES "${PROJECT_SOURCE_DIR}/script/demo_3D.json" DESTINATION script)

# Copy Manual

install(FILES "${PROJECT_SOURCE_DIR}/manual/THUNDER_v1_3_User_Guide.pdf" DESTINATION manual)
