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

#include "Symmetry.h"

using namespace arma;

#define VOL_TRANSFORM_RL(dst, src, MAT, MAT_GEN, r) \
    VOL_TRANSFORM(RL, dst, src, MAT, MAT_GEN, r)

#define VOL_TRANSFORM_FT(dst, src, MAT, MAT_GEN, r) \
    VOL_TRANSFORM(FT, dst, src, MAT, MAT_GEN, r)

#define VOL_TRANSFORM(SPACE, dst, src, MAT, MAT_GEN, r) \
[](Volume& _dst, const Volume& _src, const double _r) \
{ \
    mat33 MAT; \
    MAT_GEN; \
    VOL_TRANSFORM_MAT(SPACE, _dst, _src, MAT, _r); \
}(dst, src, r)

#define VOL_TRANSFORM_MAT_RL(dst, src, mat, r) \
    VOL_TRANSFORM_MAT(RL, dst, src, mat, r)

#define VOL_TRANSFORM_MAT_FT(dst, src, mat, r) \
    VOL_TRANSFORM_MAT(FT, dst, src, mat, r)

#define VOL_TRANSFORM_MAT(SPACE, dst, src, mat, r) \
[](Volume& _dst, const Volume& _src, const mat33 _mat, const double _r)\
{ \
    SET_0_##SPACE(_dst); \
    VOLUME_FOR_EACH_PIXEL_RL(_dst) \
    { \
        vec3 newCor = {(double)i, (double)j, (double)k}; \
        vec3 oldCor = _mat * newCor; \
        if (norm(oldCor) < _r) \
            _dst.set##SPACE(_src.getByInterpolation##SPACE(oldCor(0), \
                                                           oldCor(1), \
                                                           oldCor(2), \
                                                           LINEAR_INTERP), \
                            i, \
                            j, \
                            k); \
    } \
}(dst, src, mat, r)

void symmetryRL(Volume& dst,
                const Volume& src,
                const Symmetry& sym,
                const double r);

void symmetryFT(Volume& dst,
                const Volume& src,
                const Symmetry& sym,
                const double r);

#endif // TRANSFORMATION_H
