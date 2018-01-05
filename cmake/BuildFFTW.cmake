set(FFTW_PATH "${PROJECT_BINARY_DIR}/external/fftw")

if(SINGLE_PRECISION)
    set(libfft "thunderfftw3f")
    set(ext_conf_flags_fft --enable-shared --enable-float --prefix=${FFTW_PATH})
    if(TARGET_X86)
        set(ext_conf_flags_fft ${ext_conf_flags_fft} --enable-sse --enable-avx)
    endif(TARGET_X86)
else(SINGLE_PRECISION)
    set(libfft "thunderfftw3")
    set(ext_conf_flags_fft --enable-shared --prefix=${FFTW_PATH})
    if(TARGET_X86)
        set(ext_conf_flags_fft ${ext_conf_flags_fft} --enable-sse2 --enable-avx)
    endif(TARGET_X86)
endif(SINGLE_PRECISION)

#set(FFTW_LIB "${FFTW_PATH}/lib/${libfft}${CMAKE_SHARED_LIBRARY_SUFFIX}")
#set(FFTW_INCLUDE "${FFTW_PATH}/include")

include(externalproject)

externalproject_add(FFTW
                    SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/packages/fftw-3.3.7
                    CONFIGURE_COMMAND <SOURCE_DIR>/configure ${ext_conf_flags_fft}
                    INSTALL_DIR ${FFTW_PATH}
                    BINARY_DIR ${FFTW_PATH}/build
                    LOG_INSTALL)

                #set(BUILD_OWN_FFTW TRUE)

                #message(STATUS "FFTW_INCLUDES:     ${FFTW_INCLUDES}")
                #message(STATUS "FFTW_LIBRARIES:    ${FFTW_LIBRARIES}")
