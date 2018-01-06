set(GSL_PATH "${PROJECT_BINARY_DIR}/external/gsl")

set(libgsl "gsl")
set(ext_conf_flags_gsl --prefix=${GSL_PATH})

#set(FFTW_LIB "${FFTW_PATH}/lib/${libfft}${CMAKE_SHARED_LIBRARY_SUFFIX}")
#set(FFTW_INCLUDE "${FFTW_PATH}/include")

set(GSL_LIBRARIES ${GSL_PATH}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${libgsl}${CMAKE_SHARED_LIBRARY_SUFFIX})

include(externalproject)

externalproject_add(GSL
                    SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/packages/gsl-2.4
                    CONFIGURE_COMMAND <SOURCE_DIR>/configure ${ext_conf_flags_gsl}
                    INSTALL_DIR ${GSL_PATH}
                    BINARY_DIR ${GSL_PATH}/build
                    LOG_INSTALL)

                #set(BUILD_OWN_FFTW TRUE)

                #message(STATUS "FFTW_INCLUDES:     ${FFTW_INCLUDES}")
                #message(STATUS "FFTW_LIBRARIES:    ${FFTW_LIBRARIES}")
