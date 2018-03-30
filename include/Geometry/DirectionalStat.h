//This header file is add by huabin
#include "huabin.h"
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
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sort.h>

#include "Config.h"
#include "Macro.h"
#include "Typedef.h"
#include "Random.h"

/**
 * Probabilty Density Function of Angular Central Gaussian Distribution
 *
 * @param x   a quaternion
 * @param sig a symmetric positive definite parameter matrix
 */
double pdfACG(const dvec4& x,
              const dmat44& sig);

/**
 * Probability Density Function of Angular Central Gaussian Distribution
 *
 * paramter matrxix:
 * k0 0  0  0
 * 0  k1 0  0
 * 0  0  k1 0
 * 0  0  0  k1 
 *
 * @param x q quaterion
 * @param k0 the first paramter
 * @param k1 the second parameter
 */
double pdfACG(const dvec4& x,
              const double k0,
              const double k1);

/**
 * Sample from an Angular Central Gaussian Distribution
 *
 * @param dst the destination table
 * @param src the symmetric positive definite parameter matrix
 * @param n   the number of samples
 */
void sampleACG(dmat4& dst,
               const dmat44& src,
               const int n);

/**
 * Sample from an Angular Central Gaussian Distribution
 *
 * paramter matrix:
 * k0 0  0  0
 * 0  k1 0  0
 * 0  0  k1 0
 * 0  0  0  k1 
 *
 * @param dst the destination table
 * @param k0  the first parameter
 * @param k1  the second parameter
 * @param n   the number of samples
 */
void sampleACG(dmat4& dst,
               const double k0,
               const double k1,
               const int n);

/**
 * Sample from an Angular Central Gaussian Distribution
 *
 * parameter matrix:
 * 1 0 0 0
 * 0 k1 0 0
 * 0 0 k2 0
 * 0 0 0 k3
 *
 * @param dst the destination table
 * @param k1  the 1st parameter
 * @param k2  the 2nd parameter
 * @param k3  the 3rd parameter
 * @param n   the number of samples
 */
void sampleACG(dmat4& dst,
               const double k1,
               const double k2,
               const double k3,
               const int n);

/**
 * Paramter Matrix Inference from Data Assuming the Distribution Follows an
 * Angular Central Gaussian Distribution
 *
 * @param dst the paramter matrix
 * @param src the data
 */
void inferACG(dmat44& dst,
              const dmat4& src);

/**
 * Parameter Inference from Data Assuming the Distribution Follows an Angular
 * Central Gaussian Distribution
 *
 * @param k0  the first parameter
 * @param k1  the second paramter
 * @param src the data
 */
void inferACG(double& k0,
              double& k1,
              const dmat4& src);

void inferACG(double& k,
              const dmat4& src);

void inferACG(double& k1,
              double& k2,
              double& k3,
              const dmat4& src);

/**
 * Parameter Inference from Data Assuming the Distribution Follows an Angular
 * Central Gaussian Distribution
 *
 * @param mean the mean of ACG distribution
 * @param src  the data
 */
void inferACG(dvec4& mean,
              const dmat4& src);

/**
 * Probabilty Density Function of von Mises Distribution M(mu, kappa)
 *
 * @param x     the orientation in unit vector
 * @param mu    the mode of the von Mises distribution in unit vector
 * @param kappa the concnetration parameter of the von Mises distribution
 */
double pdfVMS(const dvec2& x,
              const dvec2& mu,
              const double k);

/**
 * Sample from von Mises Distribution M(mu, kappa), the algorithm is from Best &
 * Fisher (1979)
 *
 * @param dst   the destination table
 * @param mu    the mode of the von Mises distribution
 * @param kappa the concentration parameter of the von Mises distribution
 * @param n     number of sample
 */
void sampleVMS(dmat2& dst,
               const dvec2& mu,
               const double k,
               const double n);

void sampleVMS(dmat4& dst,
               const dvec4& mu,
               const double k,
               const double n);

/**
 * Mode and Concentration Paramter Inference from Data Assuming the Distribution
 * Follows a von Mises Distribution
 *
 * @param mu    the mode of the von Mises distribution
 * @param kappa the concentration paramter of the von Mises distribution
 * @param src    the data
 */
void inferVMS(dvec2& mu,
              double& k,
              const dmat2& src);

void inferVMS(double& k,
              const dmat2& src);

void inferVMS(dvec4& mu,
              double& k,
              const dmat4& src);

void inferVMS(double& k,
              const dmat4& src);

#endif // DIRECTIONAL_STAT_H
