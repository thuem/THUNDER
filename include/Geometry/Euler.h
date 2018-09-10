/** @file
 *  @brief Euler.h contains several functions, for operations of quaternions, converting between Euler angles, rotation matrices and unit quaternions and sampling rotation matrices from even distribution.
 *
 *  Quaternions are a number system that extends the complex numbers. Unit quaternions provide a convenient mathematical notation for representing rotations of objects in 3D. Compared to Euler angles, they are simpler to compose and aovid the problem of glimbal lock. Compared to rotation matrices, they are more compact and more efficient. Moroever, unlike Euler angles, unit quaternions do not rely on the choosing and order of the rotation axes.
 *
 *  To be noticed, Euler angles in this file follow the standard of ZXZ Euler system. In other words, Euler angle set @f$\left\{\phi, \theta, \psi\right\}@f$ stands for rotating along Z axis with @f$\phi@f$, followed by rotating along X axis with @f$\theta@f$, and followed by rotating along Z axis with @f$\psi@f$.
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
 * @brief Calculate the product of two quaternions @f$\mathbf{q_1}@f$ and @f$\mathbf{q_2}@f$.
 *
 * Assuming that @f$\mathbf{q_1} = \left(w_1, x_1, y_1, z_1\right)@f$ and @f$\mathbf{q_2} = \left(w_2, x_2, y_2, z_2\right)@f$, the product can be calculated as
 * \f[
 *   \begin{pmatrix}
 *       w_1 \\
 *       x_1 \\
 *       y_1 \\
 *       z_1
 *   \end{pmatrix}
 *   \times
 *   \begin{pmatrix}
 *       w_2 \\
 *       x_2 \\
 *       y_2 \\
 *       z_2
 *   \end{pmatrix}
 *   =
 *   \begin{pmatrix}
 *       w_{1}w_{2} - x_{1}x_{2} - y_{1}y_{2} - z_{1}z_{2} \\
 *       w_{1}x_{2} + x_{1}w_{2} + y_{1}z_{2} - z_{1}y_{2} \\
 *       w_{1}y_{2} - x_{1}z_{2} + y_{1}w_{2} + z_{1}x_{2} \\
 *       w_{1}z_{2} + x_{1}y_{2} - y_{1}x_{2} + z_{1}w_{2}
 *   \end{pmatrix}
 * \f]
 */
void quaternion_mul(dvec4& dst,      /**< [out] product, a quaternion */
                    const dvec4& a,  /**< [in]  left multiplier, @f$\mathbf{q_1}@f$ */
                    const dvec4& b   /**< [in]  right multiplier, @f$\mathbf{q_2}@f$ */
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
 *
 * @f$\mathbf{v}@f$ must be a unit vector. Output value @f$\phi@f$ ranges @f$[0, 2\pi)@f$, and @f$\theta@f$ ranges @f$[0, \pi]@f$.
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
 * @brief Calculate the unit quaternion @f$\mathbf{q}@f$ for representing the rotation, given 3 Euler angles @f$\phi@f$, @f$\theta@f$ and @f$\psi@f$.
 */
void quaternion(dvec4& dst,         /**< [out] @f$\mathbf{q}@f$ */
                const double phi,   /**< [in]  @f$\phi@f$ */
                const double theta, /**< [in]  @f$\theta@f$ */
                const double psi    /**< [in]  @f$\psi@f$ */
               );

/**
 * @brief Calculate the unit quaternion @f$\mathbf{q}@f$ for representing the rotation, given the rotation axis @f$\mathbf{r}@f$ and the rotation angle around this axis @f$\phi@f$.
 */
void quaternion(dvec4& dst,        /**< [out] @f$\mathbf{q}@f$ */
                const double phi,  /**< [in]  @f$\phi@f$ */
                const dvec3& axis  /**< [in]  @f$\mathbf{r}@f$ */
               );

/**
 * @brief Calculate the unit quaternion @f$\mathbf{q}@f$ for representing the rotation, given the rotation matrix @f$\mathbf{R}@f$.
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
 * @brief Calculate the transformation matrix @f$\mathbf{M}@f$ of reflection against a certian plane, which is represented by its normal vector @f$\mathbf{n}@f$.
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
