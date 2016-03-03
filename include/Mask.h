/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef MASK_H
#define MASK_H

#include <cmath>
#include <functional>

#include "Volume.h"
#include "Interpolation.h"

double background(const Volume& vol,
                  const double r,
                  const double ew);

double background(const Volume& volume,
                  const Volume& alpha);

void softMask(Image& img,
              const double r,
              const double ew);

void softMask(Volume& dst,
              const Volume& src,
              const double r,
              const double ew);

void softMask(Volume& dst,
              const Volume& src,
              const Volume& alpha);

void generateMask(Volume& dst,
                  const Volume& src,
                  const double densityThreshold);

void generateMask(Volume& dst,
                  const Volume& src,
                  const double densityThreshold,
                  const double extend);

void generateMask(Volume& dst,
                  const Volume& src,
                  const double densityThreshold,
                  const double extend,
                  const double ew);

#endif // MASK_H
