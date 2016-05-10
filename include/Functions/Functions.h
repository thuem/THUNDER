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
#include <numeric>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>

#include "Typedef.h"

using namespace std;

/**
 * This macros returns the nearest integer number of a.
 */
#define AROUND(a) ((int)rint(a))

/**
 * This macro returns the maximum value among a and b.
 */
#define MAX(a, b) GSL_MAX(a, b)

/**
 * This macro returns the maximum value among a, b and c.
 */
#define MAX_3(a, b, c) MAX(MAX(a, b), c)

/**
 * This macro returns the minimum value among a and b.
 */
#define MIN(a, b) GSL_MIN(a, b)

/**
 * This macro returns the minimum value among a, b and c.
 */
#define MIN_3(a, b, c) MIN(MIN(a, b), c)

/**
 * This macro returns the quadratic sum of a and b.
 */
#define QUAD(a, b) (gsl_pow_2(a) + gsl_pow_2(b))

/**
 * This macro returns the 2-norm of a and b.
 */
#define NORM(a, b) (gsl_hypot(a, b))

/**
 * This macro returns the quadratic sum of a, b and c.
 */
#define QUAD_3(a, b, c) (gsl_pow_2(a) + gsl_pow_2(b) + gsl_pow_2(c))

/**
 * This macro returns the 2-norm of a, b and c.
 */
#define NORM_3(a, b, c) (gsl_hypot3(a, b, c))

/**
 * This function calculates the cumulative summation over v.
 * @param v a vector to be cumulative added
 */
vec cumsum(const vec& v);

uvec 

/**
 * If x is peroidic and has a period of p, change x to the counterpart in [0, p)
 * and return how many periods there are between x and the counterpart in [0,
 * p).
 * @param x the period value
 * @param p the period (should be positive)
 */
int periodic(double& x,
             const double p);

/**
 * Multiplication between two quaterions.
 * @param dst result
 * @param a left multiplier
 * @param b right multiplier
 */
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

/**
 * Inverse Fourier Transform of Modified Kaiser Bessel Function, m = 2, n = 3.
 * @param r radius
 * @param a maximum radius
 * @param alpha smooth factor
 */
double MKB_RL(const double r,
              const double a,
              const double alpha);

/**
 * Estimate form of Inverse Fourier Transform of Trilinear Interpolation Function
 * @param r radius
 */
double TIK_RL(const double r);

#endif // FUNCTIONS_H
