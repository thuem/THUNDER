AUTOMAKE_OPTIONS = subdir-objects

AM_CXXFLAGS = -I${top_srcdir}/include \
			  -I${top_srcdir}/include/Error \
			  -I${top_srcdir}/include/Functions \
			  -I${top_srcdir}/include/Image \
			  -I${top_srcdir}/include/Geometry \
			  -I${top_srcdir}/include/ML

AM_CXXFLAGS += -fPIC -std=c++11 -fopenmp

noistdir = .

noist_PROGRAMS = MickeyMouseDataGen \
				 MickeyMouse \
				 InitModelGen

THUEMLIBS = ../libthuem.a

InitModelGen_SOURCES = ../casesrc/InitModelGen.cpp
InitModelGen_LDADD = ${THUEMLIBS}

MickeyMouseDataGen_SOURCES = ../casesrc/MickeyMouseDataGen.cpp
MickeyMouseDataGen_LDADD = ${THUEMLIBS}

MickeyMouse_SOURCES = ../casesrc/MickeyMouse.cpp
MickeyMouse_LDADD = ${THUEMLIBS}
