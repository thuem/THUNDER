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

#include <Eigen/Dense>

#include "THUNDERConfig.h"
#include "Macro.h"
#include "Precision.h"

#ifdef MATRIX_BOUNDARY_NO_CHECK
#define EIGEN_NO_DEBUG
#endif

using namespace Eigen;

typedef unsigned long size_t;

typedef Matrix<size_t, Dynamic, Dynamic> umat;
typedef Matrix<size_t, Dynamic, 1> uvec;

typedef Matrix<RFLOAT, Dynamic, Dynamic> mat;
typedef Matrix<RFLOAT, Dynamic, 1> vec;

typedef Matrix<double, Dynamic, Dynamic> dmat;
typedef Matrix<double, Dynamic, 1> dvec;

typedef Matrix<RFLOAT, 2, 1> vec2;
typedef Matrix<RFLOAT, 3, 1> vec3;
typedef Matrix<RFLOAT, 4, 1> vec4;

typedef Matrix<double, 2, 1> dvec2;
typedef Matrix<double, 3, 1> dvec3;
typedef Matrix<double, 4, 1> dvec4;

typedef Matrix<RFLOAT, 1, 2> rowvec2;
typedef Matrix<RFLOAT, 1, 3> rowvec3;
typedef Matrix<RFLOAT, 1, 4> rowvec4;

typedef Matrix<double, 1, 2> drowvec2;
typedef Matrix<double, 1, 3> drowvec3;
typedef Matrix<double, 1, 4> drowvec4;

typedef Matrix<RFLOAT, 2, 2> mat22;
typedef Matrix<RFLOAT, 3, 3> mat33;
typedef Matrix<RFLOAT, 4, 4> mat44;

typedef Matrix<double, 2, 2> dmat22;
typedef Matrix<double, 3, 3> dmat33;
typedef Matrix<double, 4, 4> dmat44;

typedef Matrix<size_t, Dynamic, 2> umat2;
typedef Matrix<size_t, Dynamic, 3> umat3;
typedef Matrix<size_t, Dynamic, 4> umat4;

typedef Matrix<RFLOAT, Dynamic, 2> mat2;
typedef Matrix<RFLOAT, Dynamic, 3> mat3;
typedef Matrix<RFLOAT, Dynamic, 4> mat4;

typedef Matrix<double, Dynamic, 2> dmat2;
typedef Matrix<double, Dynamic, 3> dmat3;
typedef Matrix<double, Dynamic, 4> dmat4;

#endif // TYPEDEF_H
