AUTOMAKE_OPTIONS = subdir-objects

AM_CXXFLAGS = -std=c++11

AM_CXXFLAGS += -I${top_srcdir}/include \
               -I${top_srcdir}/external/HEALPix/include

AM_CXXFLAGS += -lfftw3f -lgsl

pkglib_LIBRARIES = libthuem.a

libthuem_a_SOURCES = src/Complex.cpp

libthuem_a_SOURCES += src/Functions/Functions.cpp \
					  src/Functions/Interpolation.cpp

libthuem_a_SOURCES += src/Image/ImageBase.cpp \
					  src/Image/BMP.cpp \
					  src/Image/Image.cpp

libthuem_a_SOURCES += src/Geometry/Euler.cpp \
					  src/Geometry/Transformation.cpp

nobase_pkginclude_HEADERS = include/Complex.h

nobase_pkginclude_HEADERS += include/Functions.h \
							 include/Interpolation.h

nobase_pkginclude_HEADERS += include/ImageBase.h \
							 include/BMP.h \
							 include/Image.h

nobase_pkginclude_HEADERS += include/Euler.h \
							 include/Transformation.h
