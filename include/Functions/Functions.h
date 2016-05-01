/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cmath>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>

#define AROUND(a) ((int)rint(a))

#define MAX(a, b) GSL_MAX(a, b)

#define MAX_3(a, b, c) MAX(MAX(a, b), c)

#define MIN(a, b) GSL_MIN(a, b)

#define MIN_3(a, b, c) MIN(MIN(a, b), c)

#define QUAD(a, b) (gsl_pow_2(a) + gsl_pow_2(b))

#define NORM(a, b) (gsl_hypot(a, b))

#define QUAD_3(a, b, c) (gsl_pow_2(a) + gsl_pow_2(b) + gsl_pow_2(c))

#define NORM_3(a, b, c) (gsl_hypot3(a, b, c))

/**
 * If x is peroidic and has a period of p, change x to the counterpart in [0, p)
 * and return how many periods there are between x and the counterpart in [0,
 * p).
 * @param x the period value
 * @param p the period (should be positive)
 */
int periodic(double& x,
             const double p);

void quaternion_mul(double* dst,
                    const double* a,
                    const double* b);

/**
 * Modified Kaiser Bessel Function with m = 2 and n = 3.
 * @param r radius
 * @param a maximum radius
 * @param alpha smooth factor
 */
double MKB_FT(const double r,
              const double a,
              const double alpha);

double MKB_RL(const double r,
              const double a,
              const double alpha);
/* Inverse Fourier Transform of Modified Kaiser Bessel Function, m = 2, n = 3 */
/* Typically, a = 1.9 and alpha = 10 */

double TIK_RL(const double r);
/* Estimate form of Inverse Fourier Transform of Trilinear Interpolation Function */

#endif // FUNCTIONS_H
