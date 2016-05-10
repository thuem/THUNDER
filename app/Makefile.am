AUTOMAKE_OPTIONS = subdir-objects

AM_CXXFLAGS = -I${top_srcdir}/include \
			  -I${top_srcdir}/include/Error \
			  -I${top_srcdir}/include/Functions \
			  -I${top_srcdir}/include/Image \
			  -I${top_srcdir}/include/Geometry

AM_CXXFLAGS += -I${top_srcdir}/external/Eigen3

AM_CXXFLAGS += -fPIC -std=c++11 -fopenmp

noistdir = .

noist_PROGRAMS = Projector

THUEMLIBS = ../libthuem.a

Projector_SOURCES = ../appsrc/Projector.cpp
Projector_LDADD = ${THUEMLIBS}
