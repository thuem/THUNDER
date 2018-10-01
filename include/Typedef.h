/** @file
 *  @author Mingxu Hu
 *  @author Hongkun Yu
 *  @author Liang Qiao
 *  @version 1.4.11.180917
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR     | TIME       | VERSION       | DESCRIPTION
 *  ------     | ----       | -------       | -----------
 *  Liang Qiao | 2018/09/17 | 1.4.11.180917 | add document
 *
 *  @brief Typedef.h contains some typedef statement to use simple name representing the namespace of Eigen library.
 *
 */

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
/**
 * @brief using size_t to represent unsigned long.
 */
typedef unsigned long size_t;

/**
 * @brief using umat to represent that define a matrix(the size is unknown) with the unsigned long-type elements. 
 */
typedef Matrix<size_t, Dynamic, Dynamic> umat;

/**
 * @brief using uvec to represent that define a column-vector(the size is unknown) with the unsigned long-type elements. 
 */
typedef Matrix<size_t, Dynamic, 1> uvec;

/**
 * @brief using mat to represent that define a matrix(the size is unknown) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, Dynamic, Dynamic> mat;

/**
 * @brief using vec to represent that define a column-vector(the size is unknown) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, Dynamic, 1> vec;

/**
 * @brief using dmat to represent that define a matrix(the size is unknown) with the double-type elements. 
 */
typedef Matrix<double, Dynamic, Dynamic> dmat;

/**
 * @brief using dvec to represent that define a column-vector(the size is unknown) with the double-type elements. 
 */
typedef Matrix<double, Dynamic, 1> dvec;

/**
 * @brief using vec2 to represent that define a column-vector(@f$2\times1@f$) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, 2, 1> vec2;

/**
 * @brief using vec3 to represent that define a column-vector(@f$3\times1@f$) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, 3, 1> vec3;

/**
 * @brief using vec4 to represent that define a column-vector(@f$4\times1@f$) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, 4, 1> vec4;

/**
 * @brief using dvec2 to represent that define a column-vector(@f$2\times1@f$) with the double-type elements. 
 */
typedef Matrix<double, 2, 1> dvec2;

/**
 * @brief using dvec3 to represent that define a column-vector(@f$3\times1@f$) with the double-type elements. 
 */
typedef Matrix<double, 3, 1> dvec3;

/**
 * @brief using dvec4 to represent that define a column-vector(@f$4\times1@f$) with the double-type elements. 
 */
typedef Matrix<double, 4, 1> dvec4;

/**
 * @brief using rowvec2 to represent that define a row-vector(@f$1\times2@f$) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, 1, 2> rowvec2;

/**
 * @brief using rowvec3 to represent that define a row-vector(@f$1\times3@f$) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, 1, 3> rowvec3;

/**
 * @brief using rowvec4 to represent that define a row-vector(@f$1\times4@f$) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, 1, 4> rowvec4;

/**
 * @brief using drowvec2 to represent that define a row-vector(@f$1\times2@f$) with the double-type elements. 
 */
typedef Matrix<double, 1, 2> drowvec2;

/**
 * @brief using drowvec3 to represent that define a row-vector(@f$1\times3@f$) with the double-type elements. 
 */
typedef Matrix<double, 1, 3> drowvec3;

/**
 * @brief using drowvec4 to represent that define a row-vector(@f$1\times4@f$) with the double-type elements. 
 */
typedef Matrix<double, 1, 4> drowvec4;

/**
 * @brief using mat22 to represent that define a matrix(@f$2\times2@f$) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, 2, 2> mat22;

/**
 * @brief using mat33 to represent that define a matrix(@f$3\times3@f$) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, 3, 3> mat33;

/**
 * @brief using mat44 to represent that define a matrix(@f$4\times4@f$) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, 4, 4> mat44;

/**
 * @brief using dmat22 to represent that define a matrix(@f$2\times2@f$) with the double-type elements. 
 */
typedef Matrix<double, 2, 2> dmat22;

/**
 * @brief using dmat33 to represent that define a matrix(@f$3\times3@f$) with the double-type elements. 
 */
typedef Matrix<double, 3, 3> dmat33;

/**
 * @brief using dmat44 to represent that define a matrix(@f$4\times4@f$) with the double-type elements. 
 */
typedef Matrix<double, 4, 4> dmat44;

/**
 * @brief using umat2 to represent that define a matrix(@f$M\times2@f$, @f$M@f$ is unknown) with the unsigned long-type elements. 
 */
typedef Matrix<size_t, Dynamic, 2> umat2;

/**
 * @brief using umat3 to represent that define a matrix(@f$M\times3@f$, @f$M@f$ is unknown) with the unsigned long-type elements. 
 */
typedef Matrix<size_t, Dynamic, 3> umat3;

/**
 * @brief using umat4 to represent that define a matrix(@f$M\times4@f$, @f$M@f$ is unknown) with the unsigned long-type elements. 
 */
typedef Matrix<size_t, Dynamic, 4> umat4;

/**
 * @brief using mat2 to represent that define a matrix(@f$M\times2@f$, @f$M@f$ is unknown) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, Dynamic, 2> mat2;

/**
 * @brief using mat3 to represent that define a matrix(@f$M\times3@f$, @f$M@f$ is unknown) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, Dynamic, 3> mat3;

/**
 * @brief using mat4 to represent that define a matrix(@f$M\times4@f$, @f$M@f$ is unknown) with the RFLOAT-type elements. 
 */
typedef Matrix<RFLOAT, Dynamic, 4> mat4;

/**
 * @brief using dmat2 to represent that define a matrix(@f$M\times2@f$, @f$M@f$ is unknown) with the double-type elements. 
 */
typedef Matrix<double, Dynamic, 2> dmat2;

/**
 * @brief using dmat3 to represent that define a matrix(@f$M\times3@f$, @f$M@f$ is unknown) with the double-type elements. 
 */
typedef Matrix<double, Dynamic, 3> dmat3;

/**
 * @brief using dmat4 to represent that define a matrix(@f$M\times4@f$, @f$M@f$ is unknown) with the double-type elements. 
 */
typedef Matrix<double, Dynamic, 4> dmat4;

#endif // TYPEDEF_H
