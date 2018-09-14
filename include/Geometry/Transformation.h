/** @file
 *  @author Mingxu Hu
 *  @version 1.4.11.080913
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR    | TIME       | VERSION       | DESCRIPTION
 *  ------    | ----       | -------       | -----------
 *  Mingxu Hu | 2015/03/23 | 0.0.1.050323  | new file
 *  Xiao Long | 2018/09/13 | 1.4.11.080913 | add documentation
 *
 *  @brief Transformation.h contains several functions, for transformation of volume according to symmetry.
 *  
 *  For volumes in real space, the right transformation matrices, according to symmetry, set each voxel's value by interpolation in Fourier space. And vice versa for volumes in Fourier space.
 */


#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include <cmath>
#include <iostream>

#include "Config.h"
#include "Macro.h"
#include "Typedef.h"
#include "Precision.h"

#include "Euler.h"

#include "Functions.h"

#include "Image.h"
#include "Volume.h"

#include "Symmetry.h"

/**
 * @brief Transform a volume in real space
 * 
 * For each voxel of volume @f$dst@f$ in real space(restricted to radius @f$r@f$ in Fourier space), transform the real space coordinate into the Fourier space coordinate by transformation matrix @f$mat@f$. Then calculate the real space value by interpolation(@f$interp@f$ indicates the type of interpolation) according to its Fourier space coordinate and set transformed volume @f$src@f$.
 */
inline void VOL_TRANSFORM_MAT_RL(Volume& dst,       /**< [out] the transformed volume in real space */
                                 const Volume& src, /**< [in]  the original volume in real space */
                                 const dmat33& mat, /**< [in]  the transformation matrix */
                                 const double r,    /**< [in]  the radius in Fourier spcae, restricts the transformed volume's range */
                                 const int interp   /**< [in]  the type of interpolation */
                                )
{ 
    #pragma omp parallel for
    SET_0_RL(dst); 

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_RL(dst)
    { 
        dvec3 newCor((double)i, (double)j, (double)k);
        dvec3 oldCor = mat * newCor; 

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

/**
 * @brief Transform a volume in Fourier space
 * 
 * For each voxel of volume @f$dst@f$ in Fourier space(restricted to radius @f$r@f$ in real space), transform the Fourier space coordinate into the real space coordinate by transformation matrix @f$mat@f$. Then calculate the Fourier space value by interpolation(@f$interp@f$ indicates the type of interpolation) according to its real space coordinate and set transformed volume @f$src@f$.
 */
inline void VOL_TRANSFORM_MAT_FT(Volume& dst,       /**< [out] the transformed volume in Fourier space */
                                 const Volume& src, /**< [in]  the original volume in Fourier space */
                                 const dmat33& mat, /**< [in]  the transformation matrix */
                                 const double r,    /**< [in]  the radius in real spcae, restricts the transformed volume's range */
                                 const int interp   /**< [in]  the type of interpolation */
                                )
{ 
    #pragma omp parallel for
    SET_0_FT(dst); 

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_FT(dst)
    { 
        dvec3 newCor((double)i, (double)j, (double)k);
        dvec3 oldCor = mat * newCor; 

        if (oldCor.squaredNorm() < gsl_pow_2(r))
            dst.setFTHalf(src.getByInterpolationFT(oldCor(0),
                                                   oldCor(1),
                                                   oldCor(2),
                                                   interp),
                          i, 
                          j, 
                          k);
    }
}

/**
 * @brief Transform a volume in real space according to symmetry
 * 
 * For volume @f$dst@f$ in real space(restricted to radius @f$r@f$ in Fourier space), generate the right transformation matrices according to symmetry @f$sym@f$ for each symmetry elements. Then set the transformed volume @f$dst@f$ by @f$interp@f$ type interpolation.
 */
inline void SYMMETRIZE_RL(Volume& dst,        /**< [out] the transformed volume in real space */
                          const Volume& src,  /**< [in]  the original volume in real space */
                          const Symmetry& sym,/**< [in]  the volumes's symmetry, generates the right transformation matrices */
                          const double r,     /**< [in]  the radius in Fourier space, restricts the transformed volume's range */
                          const int interp    /**< [in]  the type of interpolation */
                         )
{
    Volume result = src.copyVolume();

    dmat33 L, R;
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

/**
 * @brief Transform a volume in Fourier space according to symmetry
 * 
 * For volume @f$dst@f$ in Fourier space(restricted to radius @f$r@f$ in real space), generate the right transformation matrices according to symmetry @f$sym@f$ for each symmetry elements. Then set the transformed volume @f$dst@f$ by @f$interp@f$ type interpolation.
 */
inline void SYMMETRIZE_FT(Volume& dst,        /**< [out] the transformed volume in Fourier space */
                          const Volume& src,  /**< [in]  the original volume in Fourier space */
                          const Symmetry& sym,/**< [in]  the volumes's symmetry, generates the right transformation matrices */
                          const double r,     /**< [in]  the radius in real space, restricts the transformed volume's range */
                          const int interp    /**< [in]  the type of interpolation */
                         )
{
    Volume result = src.copyVolume();

    dmat33 L, R;
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
