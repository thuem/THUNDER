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
#include <gsl/gsl_sort.h>

#include "Config.h"
#include "Typedef.h"

using std::sort;

/**
 * This macros returns the nearest integer number of a.
 */
#define AROUND(a) ((int)rint(a))

/**
 * This macros returns the lower nearest integer number of a.
 */
#define FLOOR(a) ((int)floor(a))

/**
 * This macros returns the upper nearest integer number of a.
 */
#define CEIL(a) ((int)ceil(a))

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
 *
 * @param v a vector to be cumulative added
 */
vec cumsum(const vec& v);

/**
 * This function sorts a vector in ascending order and stores the result by its
 * indices.
 *
 * @param v the vector to be sorted
 */
uvec index_sort_ascend(const vec& v);

/**
 * This function sorts a vector in descending order and stores the result by its
 * indices.
 *
 * @param v the vector to be sorted
 */
uvec index_sort_descend(const vec& v);

/**
 * If x is peroidic and has a period of p, change x to the counterpart in [0, p)
 * and return how many periods there are between x and the counterpart in [0,
 * p).
 *
 * @param x the period value
 * @param p the period (should be positive)
 */
int periodic(double& x,
             const double p);

/**
 * Multiplication between two quaterions.
 *
 * @param dst result
 * @param a   left multiplier
 * @param b   right multiplier
 */
void quaternion_mul(vec4& dst,
                    const vec4& a,
                    const vec4& b);

vec4 quaternion_conj(const vec4& quat);

/**
 * Modified Kaiser Bessel Function with n = 3.
 *
 * @param r     radius
 * @param a     maximum radius
 * @param alpha smooth factor
 */
double MKB_FT(const double r,
              const double a,
              const double alpha);

/**
 * Modified Kaiser Bessel Function with n = 3.
 *
 * @param r2    square of radius
 * @param a     maximum radius
 * @param alpha smooth factor
 */
double MKB_FT_R2(const double r2,
                 const double a,
                 const double alpha);

/**
 * Inverse Fourier Transform of Modified Kaiser Bessel Function with n = 3.
 *
 * @param r     radius
 * @param a     maximum radius
 * @param alpha smooth factor
 */
double MKB_RL(const double r,
              const double a,
              const double alpha);

/**
 * Inverse Fourier Transform of Modified Kaiser Bessel Function with n = 3.
 *
 * @param r2    square of radius
 * @param a     maximum radius
 * @param alpha smooth factor
 */
double MKB_RL_R2(const double r2,
                 const double a,
                 const double alpha);

/**
 * Volume of a 3D Modified Kaiser Bessel Function Blob.
 *
 * @param a     maximum radius
 * @param alpha smooth factor
 */
double MKB_BLOB_VOL(const double a,
                    const double alpha);

/**
 * Estimate form of Inverse Fourier Transform of Trilinear Interpolation Function
 *
 * @param r radius
 */
double TIK_RL(const double r);

/**
 * Estimate form of Inverse Fourier Transform of Nearest Neighbor Interpolation Function
 */
double NIK_RL(const double r);

double median(vec src,
              const int n);

/**
 * Calculcuate the Median Absolute Deviation
 *
 * @param mean the mean value
 * @param std  the standard devation
 * @param src  the input data
 * @param n    the length of the the data
 */
void stat_MAS(double& mean,
              double& std,
              vec src,
              const int n);

#endif // FUNCTIONS_H
