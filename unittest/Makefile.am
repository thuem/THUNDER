AUTOMAKE_OPTIONS = subdir-objects

AM_CXXFLAGS = -I${top_srcdir}/include \
			  -I${top_srcdir}/include/Error \
			  -I${top_srcdir}/include/Functions \
			  -I${top_srcdir}/include/Image \
			  -I${top_srcdir}/include/Geometry

AM_CXXFLAGS += -fPIC -std=c++11

noistdir = .

noist_PROGRAMS = FunctionsTest \
				 TransformationTest \
				 ImageFunctionsTest \
				 ProjectorTest \
				 ReconstructorTest \
				 ExperimentTest \
				 EulerTest \
				 SymmetryTest \
				 ParticleTest \
				 ParallelTest

THUEMLIBS = ../libthuem.a

FunctionsTest_SOURCES = ../testsrc/FunctionsTest.cpp
FunctionsTest_LDADD = ${THUEMLIBS}

TransformationTest_SOURCES = ../testsrc/TransformationTest.cpp
TransformationTest_LDADD = ${THUEMLIBS}

ImageFunctionsTest_SOURCES = ../testsrc/ImageFunctionsTest.cpp
ImageFunctionsTest_LDADD = ${THUEMLIBS}

ProjectorTest_SOURCES = ../testsrc/ProjectorTest.cpp
ProjectorTest_LDADD = ${THUEMLIBS}

ReconstructorTest_SOURCES = ../testsrc/ReconstructorTest.cpp
ReconstructorTest_LDADD = ${THUEMLIBS}

ExperimentTest_SOURCES = ../testsrc/ExperimentTest.cpp
ExperimentTest_LDADD = ${THUEMLIBS}

EulerTest_SOURCES = ../testsrc/EulerTest.cpp
EulerTest_LDADD = ${THUEMLIBS}

SymmetryTest_SOURCES = ../testsrc/SymmetryTest.cpp
SymmetryTest_LDADD = ${THUEMLIBS}

ParticleTest_SOURCES = ../testsrc/ParticleTest.cpp
ParticleTest_LDADD = ${THUEMLIBS}

ParallelTest_SOURCES = ../testsrc/ParallelTest.cpp
ParallelTest_LDADD = ${THUEMLIBS}
