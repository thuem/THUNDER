#
# Build cuthuem with interface for thuem
#

# Set output path
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Header path for core cuthunder.h
SET(CUTHUNDER_INCLUDE ${CUDA_BASE}/include)

# Interface source file
FILE(GLOB_RECURSE SOURCES ${CUDA_BASE}/interface/*.cpp)

SET(INTERFACE_INCLUDE ${CUDA_BASE}/interface)

ADD_LIBRARY(cuthuem STATIC ${SOURCES})
TARGET_INCLUDE_DIRECTORIES(cuthuem PUBLIC
    ${CUR_INCLUDE} ${INTERFACE_INCLUDE} ${CUTHUNDER_INCLUDE} ${EXTERNAL_INCLUDE})
TARGET_LINK_LIBRARIES(cuthuem cuthuemcore ${EXTERNAL_LIBS})
