#
# cuthuem: Accelerating Library for Thunder.
#

SET(CUDA_BASE ${PROJECT_SOURCE_DIR}/gpu)

configure_file(${CUDA_BASE}/config/Device.cuh.in
               ${PROJECT_BINARY_DIR}/gpu/include/Device.cuh)

include_directories(${PROJECT_BINARY_DIR}/gpu/include)

ADD_SUBDIRECTORY(include)

ADD_SUBDIRECTORY(src)

# Set interface header path
SET(INTERFACE_INCLUDE ${CUDA_BASE}/interface PARENT_SCOPE)

ADD_SUBDIRECTORY(interface)

ADD_SUBDIRECTORY(testsrc)
