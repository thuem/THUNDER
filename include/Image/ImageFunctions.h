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

#include <gsl/gsl_math.h>

#include "Image.h"
#include "Volume.h"

void translate(Image& dst,
               const Image& src,
               const double nTransCol,
               const double nTransRow);

void meshReverse(Image& img);
// In fourier space, if iCol + iRow is odd, reverse it.

void meshReverse(Volume& vol);
// In fourier space, if iCol + iRow + iSlc is odd, reverse it.

#endif // IMAGE_FUNCTIONS_H
