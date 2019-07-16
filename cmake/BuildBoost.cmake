# set(BOOST_PATH "${PROJECT_BINARY_DIR}/external/boost_1_42_0.tar.gz")

# message(STATUS "BOOST_PATH : " ${BOOST_PATH})

include(ExternalProject)

externalproject_add(BOOST
                    URL ${PROJECT_SOURCE_DIR}/external/packages/boost_1_60_0.tar.gz
                    # SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/packages/boost_1_60_0
                    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/external/boost
                    # PREFIX ${GTEST_PATH}
                    LOG_INSTALL)
