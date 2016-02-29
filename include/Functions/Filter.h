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

void bFactorFilter(Image& dst,
                   const Image& src,
                   const double bFactor);

void bFactorFilter(Volume& dst,
                   const Volume& src,
                   const double bFactor);

void lowPassFilter(Image& dst,
                   const Image& src,
                   const double thres,
                   const double ew);
/* thres -> threshold
 * ew = edgeWidth */

void lowPassFilter(Volume& dst,
                   const Volume& src,
                   const double thres,
                   const double ew);

void highPassFilter(Image& dst,
                    const Image& src,
                    const double thres,
                    const double ew);

void highPassFilter(Volume& dst,
                    const Volume& src,
                    const double thres,
                    const double ew);

/***
void fscWeightingFilter(Volume& dst,
                        const Volume& src,
                        const Vector<double>& fsc,
                        const int step = 1);
***/

#endif // FILTER_H
