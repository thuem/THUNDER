#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.6])
AC_INIT([THUEM], [0.1], [hmx13@mails.tsinghua.edu.cn])
AM_INIT_AUTOMAKE([-Wall])

AC_CONFIG_SRCDIR([src/Complex.cpp])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX([mpicxx])
AC_PROG_CC
AM_PROG_CC_C_O
AM_PROG_AR
AC_PROG_RANLIB

# Checks for libraries.
AC_CHECK_LIB([fftw3], main)
AC_CHECK_LIB([gslcblas], main)
AC_CHECK_LIB([gsl], main)
AC_CHECK_LIB([armadillo], main)
AC_CHECK_LIB([sqlite3], main)
AC_CHECK_LIB([bingham], main)
AC_CHECK_LIB([glog], main)

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_TYPE_SIZE_T

# Checks for library functions.
AC_CHECK_FUNCS([floor memset pow sqrt])

AC_CONFIG_SUBDIRS([gpu])
AC_CONFIG_FILES([Makefile
                 unittest/Makefile
                 app/Makefile
                 case/Makefile
                 case/case1/Makefile
                 case/case2/Makefile])

AC_OUTPUT
