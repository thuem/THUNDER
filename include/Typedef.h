//This header file is add by huabin
#include "huabin.h"
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

#include "Macro.h"

#ifdef MATRIX_BOUNDARY_NO_CHECK
#define EIGEN_NO_DEBUG
#endif

using namespace Eigen;

typedef unsigned long size_t;

typedef Matrix<RFLOAT, Dynamic, Dynamic> mat;
typedef Matrix<RFLOAT, Dynamic, 1> vec;

typedef Matrix<unsigned int, Dynamic, Dynamic> umat;
typedef Matrix<unsigned int, Dynamic, 1> uvec;

typedef Matrix<RFLOAT, 2, 1> vec2;
typedef Matrix<RFLOAT, 3, 1> vec3;
typedef Matrix<RFLOAT, 4, 1> vec4;

typedef Matrix<RFLOAT, 1, 2> rowvec2;
typedef Matrix<RFLOAT, 1, 3> rowvec3;
typedef Matrix<RFLOAT, 1, 4> rowvec4;

typedef Matrix<RFLOAT, 2, 2> mat22;
typedef Matrix<RFLOAT, 3, 3> mat33;
typedef Matrix<RFLOAT, 4, 4> mat44;

typedef Matrix<RFLOAT, Dynamic, 2> mat2;
typedef Matrix<RFLOAT, Dynamic, 3> mat3;
typedef Matrix<RFLOAT, Dynamic, 4> mat4;

#endif // TYPEDEF_H
