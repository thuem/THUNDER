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

#define AROUND(a) ((a - ceil(a) <= 0.5) ? ceil(a) : floor(a))

#define MAX(a, b) GSL_MAX(a, b)

#define MAX_3(a, b, c) MAX(MAX(a, b), c)

#define MIN(a, b) GSL_MIN(a, b)

#define MIN_3(a, b, c) MIN(MIN(a, b), c)

#define NORM(a, b) sqrt(gsl_pow_2(a) + gsl_pow_2(b))

#define NORM_3(a, b, c) sqrt(gsl_pow_2(a) + gsl_pow_2(b) + gsl_pow_2(c))

int periodic(double& x,
             const double p);

void normalise(gsl_vector& vec);

double MKB_FT(const double r,
              const double a,
              const double alpha);
/* Modified Kaiser Bessel Function, m = 2 */
/* Typically, a = 1.9 and alpha = 10 */

double MKB_RL(const double r,
              const double a,
              const double alpha);
/* Inverse Fourier Transform of Modified Kaiser Bessel Function, m = 2, n = 3 */
/* Typically, a = 1.9 and alpha = 10 */

double TIK_RL(const double r);
/* Estimate form of Inverse Fourier Transform of Trilinear Interpolation Function */

#endif // FUNCTIONS_H
