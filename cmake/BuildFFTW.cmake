set(FFTW_PATH "${PROJECT_BINARY_DIR}/external/fftw")

if(SINGLE_PRECISION)
    set(FFTW_LIBRARIES ${FFTW_PATH}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}fftw3f${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_PATH}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}fftw3f_threads${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(ext_conf_flags_fft --enable-threads --enable-float --prefix=${FFTW_PATH})
else(SINGLE_PRECISION)
    set(FFTW_LIBRARIES ${FFTW_PATH}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}fftw3${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_PATH}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}fftw3_threads${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(ext_conf_flags_fft --enable-threads --prefix=${FFTW_PATH})
endif(SINGLE_PRECISION)

if(ENABLE_SIMD_512)
    set(ext_conf_flags_fft "${ext_conf_flags_fft} --enable-avx512")
endif(ENABLE_SIMD_512)

if(ENABLE_SIMD_256)
    set(ext_conf_flags_fft "${ext_conf_flags_fft} --enable-avx --enable-avx2")
endif(ENABLE_SIMD_256)

message(STATUS "FFTW_FLAGS : ${ext_conf_flags_fft}")

message(STATUS "FFTW_LIBRARIES : ${FFTW_LIBRARIES}")

include(ExternalProject)

externalproject_add(FFTW
                    SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/packages/fftw-3.3.7
                    CONFIGURE_COMMAND <SOURCE_DIR>/configure ${ext_conf_flags_fft}
                    INSTALL_DIR ${FFTW_PATH}
                    BINARY_DIR ${FFTW_PATH}/build
                    LOG_INSTALL)
