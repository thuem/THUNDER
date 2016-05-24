/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef DIRECTIONAL_STAT_H
#define DIRECTIONAL_STAT_H

#include <cmath>
#include <numeric>

#include <gsl/gsl_statistics.h>
#include <gsl/gsl_math.h>

#include "Typedef.h"
#include "Random.h"

using namespace std;

/**
 * Probabilty Density Function of Angular Central Gaussian distribution.
 * @param x a quaternion
 * @param sig a symmetric positive definite parameter matrix
 */
double pdfACG(const vec4& x,
              const mat44& sig);

double pdfACG(const vec4& x,
              const double k0,
              const double k1);

void sampleACG(mat4& dst,
               const mat44& src,
               const int n);

void sampleACG(mat4& dst,
               const double k0,
               const double k1,
               const int n);

void inferACG(mat44& dst,
              const mat4& src);

void inferACG(double& k0,
              double& k1,
              const mat4& src);

#endif // DIRECTIONAL_STAT_H
