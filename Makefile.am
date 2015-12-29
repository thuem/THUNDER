AUTOMAKE_OPTIONS = subdir-objects

AM_CXXFLAGS = -std=c++11

AM_CXXFLAGS += -I${top_srcdir}/include \
               -I${top_srcdir}/external/HEALPix/include

AM_CXXFLAGS += -lfftw3f -lgsl

pkglib_LIBRARIES = libthuem.a

libthuem_a_SOURCES = src/Complex.cpp

libthuem_a_SOURCES += src/Functions/Functions.cpp \
					  src/Functions/Interpolation.cpp

libthuem_a_SOURCES += src/Image/ImageBase.cpp

nobase_pkginclude_HEADERS = include/Complex.h \
							include/ImageBase.h

nobase_pkginclude_HEADERS += include/Functions.h \
							 include/Interpolation.h
