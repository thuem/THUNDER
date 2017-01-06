/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef TYPEDEF_H
#define TYPEDEF_H

#include <gsl/gsl_complex.h>

#include <Eigen/Dense>

#include "Macro.h"

#ifdef MATRIX_BOUNDARY_NO_CHECK
#define EIGEN_NO_DEBUG
#endif

using namespace Eigen;

typedef unsigned long size_t;

typedef gsl_complex Complex;

typedef Matrix<double, Dynamic, Dynamic> mat;
typedef Matrix<double, Dynamic, 1> vec;

typedef Matrix<unsigned int, Dynamic, Dynamic> umat;
typedef Matrix<unsigned int, Dynamic, 1> uvec;

typedef Matrix<double, 2, 1> vec2;
typedef Matrix<double, 3, 1> vec3;
typedef Matrix<double, 4, 1> vec4;

typedef Matrix<double, 1, 2> rowvec2;
typedef Matrix<double, 1, 3> rowvec3;
typedef Matrix<double, 1, 4> rowvec4;

typedef Matrix<double, 2, 2> mat22;
typedef Matrix<double, 3, 3> mat33;
typedef Matrix<double, 4, 4> mat44;

typedef Matrix<double, Dynamic, 2> mat2;
typedef Matrix<double, Dynamic, 3> mat3;
typedef Matrix<double, Dynamic, 4> mat4;

#endif // TYPEDEF_H
