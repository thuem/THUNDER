/** @file
 *  @author Mingxu Hu
 *  @author Xiao Long
 *  @version 1.4.11.080917
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR    | TIME       | VERSION       | DESCRIPTION
 *  ------    | ----       | -------       | -----------
 *  Mingxu Hu | 2015/05/23 | 0.0.1.050523  | new file
 *  Xiao Long | 2018/09/14 | 1.4.11.080914 | add documentation
 *  Mingxu Hu | 2018/09/17 | 1.4.11.080917 | modify the documentation
 *
 *  @brief Transformation.h contains several functions, for transforming volumes.
 *  
 *  This file contains severl functions for transformationg volumes, in both real space and Fourier space.
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
 * @brief Transfrom a volume @f$V@f$, given the by transformation matrix @f$\mathbf{M}@f$, in real space by interpolation, and ouput the transformed volume @f$V'@f$.
 *
 * For each voxel @f$\begin{pmatrix} x \\ y \\ z \end{pmatrix}@f$ of volume @f$V@f$ in real space, it will be transformed to
 * \f[
 *   \begin{pmatrix} 
 *       x' \\
 *       y' \\
 *       z' \\
 *   \end{pmatrix}
 *   = M
 *   \begin{pmatrix}
 *       x \\
 *       y \\
 *       z \\
 *   \end{pmatrix}
 * \f]
 */
inline void VOL_TRANSFORM_MAT_RL(Volume& dst,        /**< [out] the transformed volume @f$V'@f$ */
                                 const Volume& src,  /**< [in]  the original volume @f$V@f$ */
                                 const dmat33& mat,  /**< [in]  the transformation matrix @f$M@f$ */
                                 const double r,     /**< [in]  the radius in real spcae, restricts the transformed volume's range */
                                 const int interp    /**< [in]  the type of interpolation */
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
 * @brief Transfrom a volume @f$V@f$, given the by transformation matrix @f$\mathbf{M}@f$, in Fourier space by interpolation, and ouput the transformed volume @f$V'@f$.
 *
 * For each voxel @f$\begin{pmatrix} x \\ y \\ z \end{pmatrix}@f$ of volume @f$V@f$ in real space, it will be transformed to
 * \f[
 *   \begin{pmatrix} 
 *       x' \\
 *       y' \\
 *       z' \\
 *   \end{pmatrix}
 *   = M
 *   \begin{pmatrix}
 *       x \\
 *       y \\
 *       z \\
 *   \end{pmatrix}
 * \f]
 */
inline void VOL_TRANSFORM_MAT_FT(Volume& dst,        /**< [out] the transformed volume @f$V'@f$ */
                                 const Volume& src,  /**< [in]  the original volume @f$V@f$ */
                                 const dmat33& mat,  /**< [in]  the transformation matrix @f$M@f$ */
                                 const double r,     /**< [in]  the radius in Fourier spcae, restricts the transformed volume's range */
                                 const int interp    /**< [in]  the type of interpolation */
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
 * @brief Symmetrize volume @f$V@f$ in real space, given the symmetry elements @f$\mathbf{S} = \left\{\mathbf{s_0}, \mathbf{s_1}, \dots, \mathbf{s_{N - 1}}\right\}@f$, and putput the symmetrized volume @f$V'@f$.
 * 
 * For volume @f$V@f$ in real space, transform @f$\mathbf{V}@f$ by rotation matrix @f$s_i, 0 \leq i \leq N - 1@f$, respectively. Then sum the transformed matrices up.
 */
inline void SYMMETRIZE_RL(Volume& dst,          /**< [out] the symmetrized volume @f$V'@f$ */
                          const Volume& src,    /**< [in]  the original volume @f$V@f$ */
                          const Symmetry& sym,  /**< [in]  symmetry @f$\mathbf{S}@f$ */
                          const double r,       /**< [in]  the radius in real space, restricts the transformed volume's range */
                          const int interp      /**< [in]  the type of interpolation */
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
 * @brief Symmetrize volume @f$V@f$ in Fourier space, given the symmetry elements @f$\mathbf{S} = \left\{\mathbf{s_0}, \mathbf{s_1}, \dots, \mathbf{s_{N - 1}}\right\}@f$, and putput the symmetrized volume @f$V'@f$.
 * 
 * For volume @f$V@f$ in real space, transform @f$\mathbf{V}@f$ by rotation matrix @f$s_i, 0 \leq i \leq N - 1@f$, respectively. Then sum the transformed matrices up.
 */
inline void SYMMETRIZE_FT(Volume& dst,          /**< [out] the symmetrized volume @f$V'@f$ */
                          const Volume& src,    /**< [in]  the original volume @f$V@f$ */
                          const Symmetry& sym,  /**< [in]  symmetry @f$\mathbf{S}@f$ */
                          const double r,       /**< [in]  the radius in real space, restricts the transformed volume's range */
                          const int interp      /**< [in]  the type of interpolation */
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
