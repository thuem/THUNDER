AUTOMAKE_OPTIONS = subdir-objects

AM_CXXFLAGS = -I${top_srcdir}/include \
			  -I${top_srcdir}/include/Error \
			  -I${top_srcdir}/include/Functions \
			  -I${top_srcdir}/include/Image \
			  -I${top_srcdir}/include/Geometry \
			  -I${top_srcdir}/include/ML

AM_CXXFLAGS += -I${top_srcdir}/external/Eigen3

AM_CXXFLAGS += -fPIC -std=c++11 -fopenmp

noistdir = .

noist_PROGRAMS = ModelGen \
				 DataGen \
				 Run

THUEMLIBS = ../../libthuem.a

ModelGen_SOURCES = ../../casesrc/case2/ModelGen.cpp
ModelGen_LDADD = ${THUEMLIBS}

DataGen_SOURCES = ../../casesrc/case2/DataGen.cpp
DataGen_LDADD = ${THUEMLIBS}

Run_SOURCES = ../../casesrc/case2/Run.cpp
Run_LDADD = ${THUEMLIBS}
