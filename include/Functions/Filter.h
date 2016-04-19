/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef FILTER_H
#define FILTER_H

#include <cmath>

#include "Image.h"
#include "Volume.h"

/** @brief This function performs a B-factor filtering on an image.
 *  @param dst destination image in Fourier space
 *  @param src source image in Fourier space
 *  @param bFactor B-factor in spatial frequency
 */
void bFactorFilter(Image& dst,
                   const Image& src,
                   const double bFactor);

/** @brief This function performs a B-factor filtering on a volume.
 *  @param dst destination volume in Fourier space
 *  @param src source volume in Fourier space
 *  @param bFactor B-factor in spatial frequency
 */
void bFactorFilter(Volume& dst,
                   const Volume& src,
                   const double bFactor);

/** @brief This function performs a low pass filtering on an image.
 *  @param dst destination image in Fourier space
 *  @param src source image in Fourier space
 *  @param thres threshold of spatial frequency in pixel
 *  @param ew edge width in pixel
 */
void lowPassFilter(Image& dst,
                   const Image& src,
                   const double thres,
                   const double ew);

/** @brief This function performs a low pass filtering on a volume.
 *  @param dst destination volume in Fourier space
 *  @param src source volume in Fourier space
 *  @param thres threshold of spatial frequency in pixel
 *  @param ew edge width in pixel
 */
void lowPassFilter(Volume& dst,
                   const Volume& src,
                   const double thres,
                   const double ew);

/** @brief This function performs a high pass filtering on an image.
 *  @param dst destination image in Fourier space
 *  @param src source image in Fourier space
 *  @param thres threshold of spatial frequency
 *  @param ew edge width
 */
void highPassFilter(Image& dst,
                    const Image& src,
                    const double thres,
                    const double ew);

/** @brief This function performs a high pass filtering on a volume.
 *  @param dst destination volume in Fourier space
 *  @param src source volume in Fourier space
 *  @param thres threshold of spatial frequency
 *  @param ew edge width
 */
void highPassFilter(Volume& dst,
                    const Volume& src,
                    const double thres,
                    const double ew);

/** @brief This functions performs a weighting filtering on a volume based on
 *         FSC.
 *  @param dst destination volume in Fourier space
 *  @param src source volume in Fourier space
 *  @param fsc FSC (Fourier Ring Coefficient) on which the weighting filtering
 *             based
 */
void dscWeightingFilter(Volume& dst,
                        const Volume& src,
                        const vec& fsc);
/***
void fscWeightingFilter(Volume& dst,
                        const Volume& src,
                        const Vector<double>& fsc,
                        const int step = 1);
***/

#endif // FILTER_H
