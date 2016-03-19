/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef IMAGE_FUNCTIONS_H
#define IMAGE_FUNCTIONS_H

#include <cmath>
#include <vector>

#include <gsl/gsl_math.h>

#include <armadillo>

#include "Image.h"
#include "Volume.h"
#include "Random.h"

using namespace std;
using namespace arma;

#define VOL_PAD_RL(dst, src, pf) \
    VOL_PAD(RL, dst, src, pf)

#define VOL_PAD_FT(dst, src, pf) \
    VOL_PAD(FT, dst, src, pf)

#define VOL_PAD(SP, dst, src, pf) \
    [](Volume& _dst, const Volume& _src, const int _pf) \
    { \
        _dst.alloc(_pf * _src.nColRL(), \
                   _pf * _src.nRowRL(), \
                   _pf * _src.nSlcRL(), \
                    SP##_SPACE); \
        SET_0_##SP(_dst); \
        VOLUME_FOR_EACH_PIXEL_##SP(_src) \
            _dst.set##SP(_src.get##SP(i, j, k), i, j, k); \
    }(dst, src, pf)

void translate(Image& dst,
               const Image& src,
               const double nTransCol,
               const double nTransRow);

// void meshReverse(Image& img);
/* In fourier space, if iCol + iRow is odd, reverse it. */

// void meshReverse(Volume& vol);
/* In fourier space, if iCol + iRow + iSlc is odd, reverse it. */

void bgMeanStddev(double& mean,
                  double& stddev,
                  const Image& src,
                  const double r);
/* calculate the mean and standard deviation of the background */

void removeDust(Image& img,
                const double wDust,
                const double bDust,
                const double mean,
                const double stddev);
/* remove white and black dust
 * value > mean + wDust * stddev will be replace by a draw of 
 * Gaussian(mean, stddev)
 * value < mean - kDust * stddev will be replace by a draw of 
 * Gaussian(mean, stddev) */

void normalise(Image& img,
               const double wDust,
               const double bDust,
               const double r);
/* normalise the image according to the mean and stddev of the background
 * dust points are removed according to wDust and bDust */

void extract(Image& dst,
             const Image& src,
             const int xOff,
             const int yOff);

void slice(Image& dst,
           const Volume& src,
           const int iSlc);

#endif // IMAGE_FUNCTIONS_H
