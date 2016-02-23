AUTOMAKE_OPTIONS = subdir-objects

SUBDIRS = . unittest

AM_CXXFLAGS = -std=c++11

AM_CXXFLAGS += -I${top_srcdir}/include \
			   -I${top_srcdir}/include/Error \
			   -I${top_srcdir}/include/Functions \
			   -I${top_srcdir}/include/Image \
			   -I${top_srcdir}/include/Geometry \
               -I${top_srcdir}/external/HEALPix/include

pkglib_LIBRARIES = libthuem.a

libthuem_a_SOURCES = src/Complex.cpp \
					 src/Coordinate5D.cpp \
					 src/FFT.cpp \
					 src/Projector.cpp \
					 src/Database.cpp \
					 src/Reconstructor.cpp

libthuem_a_SOURCES += src/Error/Error.cpp \
					  src/Error/Sqlite3Error.cpp

libthuem_a_SOURCES += src/Functions/Functions.cpp

libthuem_a_SOURCES += src/Image/ImageBase.cpp \
					  src/Image/BMP.cpp \
					  src/Image/Image.cpp \
					  src/Image/Volume.cpp \
					  src/Image/ImageFile.cpp \
					  src/Image/ImageFunctions.cpp

libthuem_a_SOURCES += src/Geometry/Euler.cpp \
					  src/Geometry/Transformation.cpp \
					  src/Geometry/SymmetryOperation.cpp \
					  src/Geometry/SymmetryFunctions.cpp \
					  src/Geometry/Symmetry.cpp

nobase_pkginclude_HEADERS = include/Complex.h \
							include/Coordinate5D.h \
							include/FFT.h \
							include/Projector.h \
							include/Database.h \
							include/Reconstructor.h

nobase_pkginclude_HEADERS += include/Error/Error.h \
							 include/Error/Sqlite3Error.h

nobase_pkginclude_HEADERS += include/Functions/Functions.h

nobase_pkginclude_HEADERS += include/Image/ImageBase.h \
							 include/Image/BMP.h \
							 include/Image/Image.h \
							 include/Image/Volume.h \
							 include/Image/ImageFile.h \
							 include/Image/ImageFunctions.h

nobase_pkginclude_HEADERS += include/Geometry/Euler.h \
							 include/Geometry/Transformation.h \
							 include/Geometry/SymmetryOperation.h \
							 include/Geometry/SymmetryFunctions.h \
							 include/Geometry/Symmetry.h
