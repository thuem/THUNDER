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

/**
 * This function performs a B-factor filtering on an image.
 *
 * @param dst     destination image in Fourier space
 * @param src     source image in Fourier space
 * @param bFactor B-factor in spatial frequency
 */
void bFactorFilter(Image& dst,
                   const Image& src,
                   const double bFactor);

/**
 * This function performs a B-factor filtering on a volume.
 *
 * @param dst     destination volume in Fourier space
 * @param src     source volume in Fourier space
 * @param bFactor B-factor in spatial frequency
 */
void bFactorFilter(Volume& dst,
                   const Volume& src,
                   const double bFactor);

/**
 * This function performs a low pass filtering on an image.
 * 
 * @param dst   destination image in Fourier space
 * @param src   source image in Fourier space
 * @param thres threshold of spatial frequency
 * @param ew    edge width
 */
void lowPassFilter(Image& dst,
                   const Image& src,
                   const double thres,
                   const double ew);

/**
 * This function performs a low pass filtering on a volume.
 *
 * @param dst   destination volume in Fourier space
 * @param src   source volume in Fourier space
 * @param thres threshold of spatial frequency
 * @param ew    edge width
 */
void lowPassFilter(Volume& dst,
                   const Volume& src,
                   const double thres,
                   const double ew);

/**
 * This function performs a high pass filtering on an image.
 *
 * @param dst   destination image in Fourier space
 * @param src   source image in Fourier space
 * @param thres threshold of spatial frequency
 * @param ew    edge width
 */
void highPassFilter(Image& dst,
                    const Image& src,
                    const double thres,
                    const double ew);

/**
 * This function performs a high pass filtering on a volume.
 *
 * @param dst   destination volume in Fourier space
 * @param src   source volume in Fourier space
 * @param thres threshold of spatial frequency
 * @param ew    edge width
 */
void highPassFilter(Volume& dst,
                    const Volume& src,
                    const double thres,
                    const double ew);

/**
 * This functions performs a weighting filtering on a volume based on FSC.
 *
 * @param dst destination volume in Fourier space
 * @param src source volume in Fourier space
 * @param fsc FSC (Fourier Ring Coefficient) on which the weighting filtering
 *            based
 */
void fscWeightingFilter(Volume& dst,
                        const Volume& src,
                        const vec& fsc);

#endif // FILTER_H
