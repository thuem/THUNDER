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

void translate(Image& dst,
               const Image& src,
               const double nTransCol,
               const double nTransRow);

void meshReverse(Image& img);
/* In fourier space, if iCol + iRow is odd, reverse it. */

void meshReverse(Volume& vol);
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

#endif // IMAGE_FUNCTIONS_H
