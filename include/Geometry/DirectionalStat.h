/** @file
 *  @author Mingxu Hu
 *  @author Zhao Wang
 *  @version 1.4.11.080927
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR    | TIME       | VERSION       | DESCRIPTION
 *  ------    | ----       | -------       | -----------
 *  Mingxu Hu | 2015/03/23 | 0.0.1.050323  | new file
 *  Zhao Wang | 2018/09/27 | 1.4.11.080927 | add documentation 
 *  
 *  @brief DirectionStat.h contains several functions to carry out computations in Angular Central Gaussian and von Mises Distribution respectively.
 *
 *  The suffix "ACG" and "VMS" in the function names denotes two modes of distributions, "Angular Central Gaussian Distribution" and "von Mises Distribution". The prefix represents the computation to be executed. "pdf" returns the probability density function; "sample" implements sampling operation; "infer" gives the parameter matrix inference.
 */


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
#include "Precision.h"
#include "Random.h"
#include "Functions.h"

/**
 * @brief Calculate the probability density function of angular central Gaussian distribution.
 */
double pdfACG(const dvec4& x,       /**< [in] a quaternion */
              const dmat44& sig     /**< [in] a symmetric positive definite parameter matrix */
              );

/**
 * @brief Calculate the probability density function of angular central Gaussian distribution. The parameter matrix of this ACG distribution is
 * \f[
 *   \begin{pmatrix}
 *       k0 & 0  & 0  & 0 \\
 *       0  & k1 & 0  & 0 \\
 *       0  & 0  & k1 & 0 \\
 *       0  & 0  & 0  & k1 
 *   \end{pmatrix}
 * \f]
 */
double pdfACG(const dvec4& x,      /**< [in] a quaternion */
              const double k0,     /**< [in] first parameter of a positive-definite matrix */
              const double k1      /**< [in] second parameter of a positive-definite matrix */
              );

/**
 * @brief Sample from an angular central Gaussian distribution.
 */
void sampleACG(dmat4& dst,         /**< [in] the destination table */
               const dmat44& src,  /**< [in] the symmetric positive definite parameter matrix */
               const int n         /**< [in] the number of samples */
               );

/**
 * @brief Sample from an angular central Gaussian distribution. The parameter matrix of this ACG distribution is
 * \f[
 *   \begin{pmatrix}
 *       k0 & 0  & 0  & 0 \\
 *       0  & k1 & 0  & 0 \\
 *       0  & 0  & k1 & 0 \\
 *       0  & 0  & 0  & k1 
 *   \end{pmatrix}
 * \f]
 */
void sampleACG(dmat4& dst,         /**< [in] the destination table */
               const double k0,    /**< [in] first parameter of a positive-definite matrix */
               const double k1,    /**< [in] second parameter of a positive-definite matrix */
               const int n         /**< [in] the number of samples */
               );

/**
 * @brief Sample from an angular central Gaussian distribution. The parameter matrix of this ACG distribution is
 * \f[
 *   \begin{pmatrix}
 *       1 & 0  & 0  & 0 \\
 *       0 & k1 & 0  & 0 \\
 *       0 & 0  & k2 & 0 \\
 *       0 & 0  & 0  & k3 
 *   \end{pmatrix}
 * \f]
 */
void sampleACG(dmat4& dst,        /**< [in] the destination table */
               const double k1,   /**< [in] first parameter of a positive-definite matrix */
               const double k2,   /**< [in] second parameter of a positive-definite matrix */
               const double k3,   /**< [in] third parameter of a positive-definite matrix */
               const int n        /**< [in] the number of samples */
               );

/**
 * @brief Calculate the parameter matrix inference from source data assuming the distribution follows an angular central Gaussian distribution.
 */
void inferACG(dmat44& dst,       /**< [in] the parameter matrix */
              const dmat4& src   /**< [in] the given data */
              );

/**
 * @brief Calculate the parameter matrix inference from source data assuming the distribution follows an angular central Gaussian distribution.
 */
void inferACG(double& k0,       /**< [in] first parameter of a positive-definite matrix */
              double& k1,       /**< [in] second parameter of a positive-definite matrix */
              const dmat4& src  /**< [in] the given data */
              );

/**
 * @brief Calculate the parameter matrix inference from source data assuming the distribution follows an angular central Gaussian distribution.
 */
void inferACG(double& k,        /**< [in]  */
              const dmat4& src  /**< [in] the given data */
              );

/**
 * @brief Calculate the parameter matrix inference from source data assuming the distribution follows an angular central Gaussian distribution.
 */
void inferACG(double& k1,       /**< [in] first parameter of a positive-definite matrix */
              double& k2,       /**< [in] second parameter of a positive-definite matrix */
              double& k3,       /**< [in] third parameter of a positive-definite matrix */
              const dmat4& src  /**< [in] the given data */
              );

/**
 * @brief Calculate the parameter matrix inference from source data assuming the distribution follows an angular central Gaussian distribution.
 */
void inferACG(dvec4& mean,      /**< [in] the mean of ACG distribution */
              const dmat4& src  /**< [in] the given data */
              );

/**
 * @brief Calculate the probability density function of von Mises Distribution M(mu, kappa).
 */
double pdfVMS(const dvec2& x,   /**< [in] the orientation in unit vector */
              const dvec2& mu,  /**< [in] the mode of the von Mises distribution in unit vector */
              const double k    /**< [in] the concentration parameter of the von Mises distribution */
              );

/**
 * @brief Sample from von Mises Distribution M(mu, kappa), the algorithm is from Best & Fisher (1979).
 */
void sampleVMS(dmat2& dst,      /**< [in] the destination table */
               const dvec2& mu, /**< [in] the mode of the von Mises distribution */
               const double k,  /**< [in] the concentration parameter of the von Mises distribution */
               const double n   /**< [in] the number of samples */
               );

/**
 * @brief Sample from von Mises Distribution M(mu, kappa), the algorithm is from Best & Fisher (1979).
 */
void sampleVMS(dmat4& dst,      /**< [in] the destination table */
               const dvec4& mu, /**< [in] the mode of the von Mises distribution */
               const double k,  /**< [in] the concentration parameter of the von Mises distribution */
               const double n   /**< [in] the number of samples */
               );

/**
 * @brief Calculate the mode and concentration parameter inference from the given data assuming the distribution follows a von Mises Distribution.
 */
void inferVMS(dvec2& mu,        /**< [in] the mode of the von Mises distribution */
              double& k,        /**< [in] the concentration parameter of the von Mises distribution */
              const dmat2& src  /**< [in] the given data */
              );

/**
 * @brief Calculate the mode and concentration parameter inference from the given data assuming the distribution follows a von Mises Distribution.
 */
void inferVMS(double& k,        /**< [in] the concentration parameter of the von Mises distribution */
              const dmat2& src  /**< [in] the given data */
              );

/**
 * @brief Calculate the mode and concentration parameter inference from the given data assuming the distribution follows a von Mises Distribution.
 */
void inferVMS(dvec4& mu,        /**< [in] the mode of the von Mises distribution */
              double& k,        /**< [in] the concentration parameter of the von Mises distribution */
              const dmat4& src  /**< [in] the given data */
              );

/**
 * @brief Calculate the mode and concentration parameter inference from the given data assuming the distribution follows a von Mises Distribution.
 */
void inferVMS(double& k,        /**< [in] the concentration parameter of the von Mises distribution */
              const dmat4& src  /**< [in] the given data */
              );

#endif // DIRECTIONAL_STAT_H
