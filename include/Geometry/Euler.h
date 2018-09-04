/** @file
 *  @brief some description about Euler.h
 *
 *  Details about Euler.h
 */

#ifndef EULER_H
#define EULER_H

#include <cmath>

#include <gsl/gsl_math.h>

#include "Macro.h"
#include "Typedef.h"
#include "Precision.h"
#include "Random.h"
#include "Functions.h"

/**
 * @brief Calculate the product of two quaternions.
 */
void quaternion_mul(dvec4& dst,      /**< [out] product, a quaternion */
                    const dvec4& a,  /**< [in]  left multiplier, quaternion */
                    const dvec4& b   /**< [in]  right multiplier, quaternion */
                   );
/**
 * @brief Calculate the conjugate quaternion of a quaternion.
 *
 * @return the conjugate quaternion
 */
dvec4 quaternion_conj(const dvec4& quat /**< [in] a quaternion */
                     );

/**
 * @brief Calculate @f$\phi@f$ and @f$\theta@f$ given a certain direction @f$\mathbf{v}@f$.
 */
void angle(double& phi,     /**< [out] @f$\phi@f$ */
           double& theta,   /**< [out] @f$\theta@f$ */
           const dvec3& src /**< [in]  @f$\mathbf{v}@f$ */
          );

/**
 * @brief Calculate @f$\phi@f$, @f$\theta@f$ and @f$\psi@f$ of the rotation represented by the rotation matrix @f$\mathbf{R}@f$.
 */
void angle(double& phi,      /**< [out] @f$\phi@f$ */
           double& theta,    /**< [out] @f$\theta@f$ */
           double& psi,      /**< [out] @f$\psi@f$ */
           const dmat33& src /**< [in]  @f$\mathbf{R}@f$ */
          );

/**
 * @brief Calculate @f$\phi@f$, @f$\theta@f$ and @f$\psi@f$ of the rotation represented by the quaternion @f$\mathbf{q}@f$.
 */
void angle(double& phi,     /**< [out] @f$\phi@f$ */
           double& theta,   /**< [out] @f$\theta@f$ */
           double& psi,     /**< [out] @f$\psi@f$ */
           const dvec4& src /**< [in]  @f$\mathbf{q}@f$ */
          );

/**
 * @brief Calculate the quaternion @f$\mathbf{q}@f$ for representing the rotation, given 3 Euler angles @f$\phi@f$, @f$\theta@f$ and @f$\psi@f$.
 */
void quaternion(dvec4& dst,         /**< [out] @f$\mathbf{q}@f$ */
                const double phi,   /**< [in]  @f$\phi@f$ */
                const double theta, /**< [in]  @f$\theta@f$ */
                const double psi    /**< [in]  @f$\psi@f$ */
               );

/**
 * @brief Calculate the quaternion @f$\mathbf{q}@f$ for representing the rotation, given the rotation axis @f$\mathbf{r}@f$ and the rotation angle around this axis @f$\phi@f$.
 */
void quaternion(dvec4& dst,        /**< [out] @f$\mathbf{q}@f$ */
                const double phi,  /**< [in]  @f$\phi@f$ */
                const dvec3& axis  /**< [in]  @f$\mathbf{r}@f$ */
               );

/**
 * @brief Calculate the quaternion @f$\mathbf{q}@f$ for representing the rotation, given the rotation matrix @f$\mathbf{R}@f$.
 */
void quaternion(dvec4& dst,       /**< [out] @f$\mathbf{q}@f$ */
                const dmat33& src /**< [in]  @f$\mathbf{R}@f$ */
               );

/**
 * @brief Calculate the rotation matrix (2D) @f$\mathbf{R}@f$, which rotates the unit vector @f$\mathbf{v_0} = \left\{1, 0\right\}@f$ to the given unit vector @f$\mathbf{v}@f$.
 */
void rotate2D(dmat22& dst,     /**< [out] @f$\mathbf{R}@f$ */
              const dvec2& vec /**< [in]  @f$\mathbf{v}@f$ */
             );

/**
 * @brief Calculate the rotation matrix (2D) @f$\mathbf{R}@f$, given the rotation angle @f$\phi@f$.
 */
void rotate2D(dmat22& dst,     /**< [out] @f$\mathbf{R}@f$ */
              const double phi /**< [in]  @f$\phi@f$ */
             );

/**
 * @brief Caclulate the unit direction vector @f$\mathbf{v}@f$, given the rotation angle @f$\phi@f$ and @f$\theta@f$.
 */
void direction(dvec3& dst,        /**< [out] @f$\mathbf{v}@f$ */
               const double phi,  /**< [in]  @f$\phi@f$ */
               const double theta /**< [in]  @f$\theta@f$ */
              );

/**
 * @brief Caclulate the rotation matrix @f$\mathbf{R}@f$, given the rotation angle @f$\phi@f$, @f$\theta@f$ and @f$\psi@f$.
 */
void rotate3D(dmat33& dst,        /**< [out] @f$\mathbf{R}@f$ */
              const double phi,   /**< [in]  @f$\phi@f$ */
              const double theta, /**< [in]  @f$\theta@f$ */
              const double psi    /**< [in]  @f$\psi@f$ */
             );

/**
 * @brief Calculate the rotation matrix @f$\mathbf{R}@f$, given the unit quaternion @f$\mathbf{q}@f$ which represents this rotation.
 */
void rotate3D(dmat33& dst,     /**< [out] @f$\mathbf{R}@f$ */
              const dvec4& src /**< [in]  @f$\mathbf{q}@f$ */
             );

/**
 * @brief Calculate the rotation matrix @f$\mathbf{R}@f$ which represents the rotation along X-axis with rotation angle @f$\phi@f$.
 */
void rotate3DX(dmat33& dst,     /**< [out] @f$\mathbf{R}@f$ */
               const double phi /**< [in]  @f$\phi@f$ */
              );

/**
 * @brief Calculate the rotation matrix @f$\mathbf{R}@f$ which represents the rotation along Y-axis with rotation angle @f$\phi@f$.
 */
void rotate3DY(dmat33& dst,     /**< [out] @f$\mathbf{R}@f$ */
               const double phi /**< [in]  @f$\phi@f$ */
              );

/**
 * @brief Calculate the rotation matrix @f$\mathbf{R}@f$ which represents the rotation along Z-axis with rotation angle @f$\phi@f$.
 */
void rotate3DZ(dmat33& dst,     /**< [out] @f$\mathbf{R}@f$ */
               const double phi /**< [in]  @f$\phi@f$ */
              );

/**
 * @brief Calculate the rotation matrix @f$\mathbf{R}@f$ which aligns a direction vector @f$\mathbf{v}@f$ to Z-axis.
 */
void alignZ(dmat33& dst,     /**< [out] @f$\mathbf{R}@f$ */
            const dvec3& vec /**< [in]  @f$\mathbf{v}@f$ */
           );

/**
 * @brief Calculate the rotation matrix @f$\mathbf{R}@f$ which represents the rotation along the axis @f$\mathbf{v}@f$ with rotation angle @f$\phi@f$.
 */
void rotate3D(dmat33& dst,      /**< [out] @f$\mathbf{R}@f$ */
              const double phi, /**< [in]  @f$\phi@f$ */
              const dvec3& axis /**< [in]  @f$\mathbf{v}@f$ */
             );

/**
 * @brief Calculate the transformation matrix @f$\mathbbf{M}@f$ of reflection against a certian plane, which is represented by its normal vector @f$\mathbf{n}@f$.
 */
void reflect3D(dmat33& dst,       /**< [out] @f$\mathbf{M}@f$ */
               const dvec3& plane /**< [in]  @f$\mathbf{n}@f$ */
              );

/**
 * @brief Calculate the two quaternions @f$\mathbf{q_s}@f$ and @f$\mathbf{q_t}@f$, which represent swing and twist along axis @f$\mathbf{v}@f$ respectively, representing the rotation represented by quaternion @f$\mathbf{q}@f$.
 */
void swingTwist(dvec4& swing,     /**< [out] @f$\mathbf{q_s}@f$ */
                dvec4& twist,     /**< [out] @f$\mathbf{q_t}@f$ */
                const dvec4& src, /**< [in]  @f$\mathbf{q}@f$ */
                const dvec3& vec  /**< [in]  @f$\mathbf{v}@f$ */
               );

/**
 * @brief Sample a 2D rotation matrix @f$\mathbf{R}@f$ from even distribution.
 */
void randRotate2D(dmat22& rot /**< [out] @f$\mathbf{R}@f$ */
                 );

/**
 * @brief Sample a 3D rotation matrix @f$\mathbf{R}@f$ from even distribution.
 */
void randRotate3D(dmat33& rot /**< [out] @f$\mathbf{R}@f$ */
                 );

#endif // EULER_H 
