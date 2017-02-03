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

#include "Config.h"
#include "Macro.h"
#include "Typedef.h"

#include "Euler.h"

#include "Functions.h"

#include "Image.h"
#include "Volume.h"

#include "Symmetry.h"

inline void VOL_TRANSFORM_MAT_RL(Volume& dst, 
                                 const Volume& src, 
                                 const mat33& mat, 
                                 const double r,
                                 const int interp)
{ 
    #pragma omp parallel for
    SET_0_RL(dst); 

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_RL(dst)
    { 
        vec3 newCor((double)i, (double)j, (double)k);
        vec3 oldCor = mat * newCor; 

        if (oldCor.squaredNorm() < gsl_pow_2(r))
            dst.setRL(src.getByInterpolationRL(oldCor(0),
                                               oldCor(1),
                                               oldCor(2),
                                               interp),
                      i, 
                      j, 
                      k); 
    } 
}

inline void VOL_TRANSFORM_MAT_FT(Volume& dst, 
                                 const Volume& src, 
                                 const mat33& mat, 
                                 const double r,
                                 const int interp)
{ 
    #pragma omp parallel for
    SET_0_FT(dst); 

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_FT(dst)
    { 
        vec3 newCor((double)i, (double)j, (double)k);
        vec3 oldCor = mat * newCor; 

        if (oldCor.squaredNorm() < gsl_pow_2(r))
            dst.setFT(src.getByInterpolationFT(oldCor(0),
                                               oldCor(1),
                                               oldCor(2),
                                               interp),
                      i, 
                      j, 
                      k);
    }
}

inline void SYMMETRIZE_RL(Volume& dst,
                          const Volume& src,
                          const Symmetry& sym,
                          const double r,
                          const int interp)
{
    Volume result = src.copyVolume();

    mat33 L, R;
    Volume se(src.nColRL(), src.nRowRL(), src.nSlcRL(), RL_SPACE);

    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        sym.get(L, R, i);

        VOL_TRANSFORM_MAT_RL(se, src, R, r, interp);

        #pragma omp parallel for
        ADD_RL(result, se);
    }

    dst.swap(result);
}

inline void SYMMETRIZE_FT(Volume& dst,
                          const Volume& src,
                          const Symmetry& sym,
                          const double r,
                          const int interp)
{
    Volume result = src.copyVolume();

    mat33 L, R;
    Volume se(src.nColRL(), src.nRowRL(), src.nSlcRL(), FT_SPACE);

    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        sym.get(L, R, i);

        VOL_TRANSFORM_MAT_FT(se, src, R, r, interp);

        #pragma omp parallel for
        ADD_FT(result, se);
    }

    dst.swap(result);
}

#endif // TRANSFORMATION_H
