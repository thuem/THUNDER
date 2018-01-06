set(GSL_PATH "${PROJECT_BINARY_DIR}/external/gsl")

set(ext_conf_flags_gsl --prefix=${GSL_PATH})

set(GSL_LIBRARIES ${GSL_PATH}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gsl${CMAKE_STATIC_LIBRARY_SUFFIX})
set(GSL_LIBRARIES ${GSL_LIBRARIES} ${GSL_PATH}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gslcblas${CMAKE_STATIC_LIBRARY_SUFFIX})

message(STATUS "GSL_LIBRARIES : ${GSL_LIBRARIES}")

include(ExternalProject)

externalproject_add(GSL
                    SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/packages/gsl-2.4
                    CONFIGURE_COMMAND <SOURCE_DIR>/configure ${ext_conf_flags_gsl}
                    INSTALL_DIR ${GSL_PATH}
                    BINARY_DIR ${GSL_PATH}/build
                    LOG_INSTALL)
