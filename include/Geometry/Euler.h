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
void quaternion_mul(dvec4& dst /**< [out] product, a quaternion */,
                    const dvec4& a /**< [in] left multiplier, quaternion */,
                    const dvec4& b /**< [in] right multiplier, quaternion */);

/**
 * @brief Calculate the conjugate quaternion of a quaternion.
 *
 * @return the conjugate quaternion
 */
dvec4 quaternion_conj(const dvec4& quat /**< [in] a quaternion */);

/**
 * @brief Calculate @f$\phi@f$ and @f$\theta@f$ given a certain direction @f$\mathbf{v}@f$.
 */
void angle(double& phi /**< [out] @f$\phi@f$ */,
           double& theta /**< [out] @f$\theta@f$ */,
           const dvec3& src /**< [in] @f$\mathbf{v}@f$ */);

/**
 * @brief Calculate @f$\phi@f$, @f$\theta@f$ and @f$\psi@f$ of the rotation represented by the rotation matrix @f$\mathbf{R}@f$.
 */
void angle(double& phi /**< [out] @f$\phi@f$ */,
           double& theta /**< [out] @f$\theta@f$ */,
           double& psi /**< [out] @f$\psi@f$ */,
           const dmat33& src /**< [in] @f$\mathbf{R}@f$ */);

/**
 * @brief Calculate @f$\phi@f$, @f$\theta@f$ and @f$\psi@f$ of the rotation represented by the quaternion @f$\mathbf{q}@f$.
 */
void angle(double& phi /**< [out] @f$\phi@f$ */,
           double& theta /**< [out] @f$\theta@f$ */,
           double& psi /**< [out] @f$\psi@f$ */,
           const dvec4& src /**< [in] @f$\mathbf{q}@f$ */);

/**
 * @brief Calculate the quaternion @f$\mathbf{q}@f$ for representing the rotation, given 3 Euler angles @f$\phi@f$, @f$\theta@f$ and @f$\psi@f$.
 */
void quaternion(dvec4& dst /**< [out] @f$\mathbf{q}@f$ */,
                const double phi /**< [in] @f$\phi@f$ */,
                const double theta /**< [in] @f$\theta@f$ */,
                const double psi /**< [in] @f$\psi@f$ */);

/**
 * @brief Calculate the quaternion @f$\mathbf{q}@f$ for representing the rotation, given the rotation axis @f$\mathbf{r}@f$ and the rotation angle around this axis @f$\phi@f$.
 */
void quaternion(dvec4& dst /**< [out] @f$\mathbf{q}@f$ */,
                const double phi /**< [in] @f$\phi@f$ */,
                const dvec3& axis /**< [in] @f$\mathbf{r}@f$ */);

/**
 * @brief Calculate the quaternion @f$\mathbf{q}@f$ for representing the rotation, given the rotation matrix @f$\mathbf{R}@f$.
 */
void quaternion(dvec4& dst /**< [out] @f$\mathbf{q}@f$ */,
                const dmat33& src /**< [in] @f$\mathbf{R}@f$ */);

/**
 * @brief Calculate the rotation matrix (2D) @f$\mathbf{R}@f$, which rotates the unit vector @f$\mathbf{v_0} = \left\{1, 0\right\}@f$ to the given unit vector @f$\mathbf{v}@f$.
 *
 * @param dst the rotation matrix
 * @param vec the unit vector
 */
void rotate2D(dmat22& dst /**< [out] @f$\mathbf{R}@f$ */,
              const dvec2& vec /**< [in] @f$\mathbf{v}@f$ */);

/**
 * @brief Calculate the rotation matrix (2D) @f$\mathbf{R}@f$, given the rotation angle @f$\phi@f$.
 */
void rotate2D(dmat22& dst /**< [out] @f$\mathbf{R}@f$ */,
              const double phi /**< [in] @f$\phi@f$ */);

/**
 * @brief Caclulate the unit direction vector @f$\mathbf{v}@f$, given the rotation angle @f$\phi@f$ and @f$\theta@f$.
 */
void direction(dvec3& dst /**< [out] @f$\mathbf{v}@f$ */,
               const double phi /**< [in] @f$\phi@f$ */,
               const double theta /**< [in] @f$\theta@f$ */);

/**
 * @brief Caclulate the rotation matrix @f$\mathbf{R}@f$, given the rotation angle @f$\phi@f$, @f$\theta@f$ and @f$\psi@f$.
 */
void rotate3D(dmat33& dst /**< [out] @f$\mathbf{R}@f$ */,
              const double phi /**< [in] @f$\phi@f$ */,
              const double theta /**< [in] @f$\theta@f$ */,
              const double psi /**< [in] @f$\psi@f$ */);

/**
 * @brief Calculate the rotation matrix @f$\mathbf{R}@f$, given the unit quaternion @f$\mathbf{q}@f$ which represents this rotation.
 */
void rotate3D(dmat33& dst /**< [out] @f$\mathbf{R}@f$ */,
              const dvec4& src /**< [in] @f$\mathbf{q}@f$ */);

/**
 * This function calculates the rotation matrix of rotation along X-axis of phi.
 *
 * @param dst the rotation matrix
 * @param phi phi
 */
void rotate3DX(dmat33& dst, const double phi);

/**
 * This function calculates the rotation matrix of rotation along Y-axis of phi.
 *
 * @param dst the rotation matrix
 * @param phi phi
 */
void rotate3DY(dmat33& dst, const double phi);

/**
 * This function calculates the rotation matrix of rotation along Z-axis of phi.
 *
 * @param dst the rotation matrix
 * @param phi phi
 */
void rotate3DZ(dmat33& dst, const double phi);

/**
 * This function calculates the rotation matrix for aligning a direction vector
 * to Z-axis.
 *
 * @param dst the rotation matrix
 * @param vec the direction vector
 */
void alignZ(dmat33& dst,
            const dvec3& vec);

/**
 * This function calculates the rotation matrix of rotation along a certain axis
 * (X, Y or Z) of phi.
 *
 * @param dst  the rotation matrix
 * @param axis a character indicating which axis the rotation is along
 */
void rotate3D(dmat33& dst,
              const double phi,
              const char axis);

/**
 * This function calculates the rotation matrix of rotation along a certain axis
 * given by a direction vector of phi.
 *
 * @param dst  the rotation matrix
 * @param phi  phi
 * @param axis the direction vector indicating the axis
 */
void rotate3D(dmat33& dst,
              const double phi,
              const dvec3& axis);

/**
 * This function calculates the transformation matrix of reflection against a
 * certain plane given by its normal vector.
 *
 * @param dst   the rotation matrix
 * @param plane the normal vector the reflection plane
 */
void reflect3D(dmat33& dst,
               const dvec3& plane);

/**
 * This function calculates the singular matrix of translation of a certain
 * vector.
 *
 * @param dst the singular matrix
 * @param vec the translation vector
 */
void translate3D(mat44& dst,
                 const dvec3& vec);

/**
 * This function calculates the transformation matrix of scaling.
 *
 * @param dst the transformation matrix
 * @param vec a 3-vector of which vec[0] indicates the scale factor along X
 *            axis, vec[1] indicates the scale factor along Y axis and vec[2]
 *            indicates the scale factor along Z axis
 */
void scale3D(dmat33& dst,
             const dvec3& vec);

void swingTwist(dvec4& swing,
                dvec4& twist,
                const dvec4& src,
                const dvec3& vec);

void randDirection(dvec2& dir);

/**
 * This function generates a random unit quaternion.
 */
//void randQuaternion(dvec4& quat);

void randRotate2D(dmat22& rot);

void randQuaternion(dvec4& quat);

/**
 * This function generates a random 3D rotation matrix.
 */
void randRotate3D(dmat33& rot);

#endif // EULER_H 
