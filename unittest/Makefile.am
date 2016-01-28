AUTOMAKE_OPTIONS = subdir-objects

AM_CXXFLAGS = -I${top_srcdir}/include \
			  -I${top_srcdir}/include/Error \
			  -I${top_srcdir}/include/Functions \
			  -I${top_srcdir}/include/Image \
			  -I${top_srcdir}/include/Geometry \
			  -I${top_srcdir}/external/HEALPix/include

AM_CXXFLAGS += -fPIC -std=c++11

AM_CXXFLAGS += -lfftw3 -lgsl -lcblas

noistdir = .

noist_PROGRAMS = FunctionsTest \
				 ImageFunctionsTest \
				 ProjectorTest

THUEMLIBS = ../libthuem.a

FunctionsTest_SOURCES = ../testsrc/FunctionsTest.cpp
FunctionsTest_LDADD = ${THUEMLIBS}

ImageFunctionsTest_SOURCES = ../testsrc/ImageFunctionsTest.cpp
ImageFunctionsTest_LDADD = ${THUEMLIBS}

ProjectorTest_SOURCES = ../testsrc/ProjectorTest.cpp
ProjectorTest_LDADD = ${THUEMLIBS}
