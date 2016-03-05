/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include <cmath>
#include <iostream>

#include <armadillo>

#include "Macro.h"
#include "Typedef.h"
#include "Error.h"

#include "Euler.h"

#include "Functions.h"

#include "Image.h"
#include "Volume.h"

using namespace arma;

#define VOL_TRANSFORM_RL(dst, src, MAT, MAT_GEN, r) \
[](Volume& _dst, const Volume& _src, const double _r)\
{ \
    mat33 MAT; \
    MAT_GEN; \
    VOLUME_FOR_EACH_PIXEL_RL(_dst) \
    { \
        vec3 newCor = {i, j, k}; \
        vec3 oldCor = (MAT) * newCor; \
        if (norm(oldCor) < _r) \
            _dst.setRL(_src.getByInterpolationRL(oldCor(0), \
                                                 oldCor(1), \
                                                 oldCor(2), \
                                                 LINEAR_INTERP), \
                      i, \
                      j, \
                      k); \
        else \
            _dst.setRL(0, i, j, k); \
    } \
}(dst, src, r)

void alignZ(mat33& dst,
            const vec3& vec);
// This function returns a 3x3 matrix.
// This matrix can align vec to Z axis.

void rotate3D(mat33& dst,
              const double phi,
              const char axis);
// This function returns a 3x3 matrix.
// X -> rotate around X axis
// Y -> rotate around Y axis
// Z -> rotate around Z axis

void rotate3D(mat33& dst,
              const double phi,
              const vec3& axis);
// This function returns a 3x3 matrix.
// This matrix represents a rotation around the axis given.

void reflect3D(mat33& dst,
               const vec3& plane);
// This function returns a 3x3 matrix.
// This matrix represents a reflection of the plane given.

void translate3D(mat44& dst,
                 const vec3& vec);
// This function returns a 4x4 matrix.
// This matrix represents a translation of the vec given.

void scale3D(mat33& dst,
             const vec3& vec);
// This function returns a 3x3 matrix.
// This matrix represents a scaling.
// scale along X -> vec[0]
// scale along Y -> vec[1]
// scale along Z -> vec[2]

#endif // TRANSFORMATION_H
