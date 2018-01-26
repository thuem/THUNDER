//This header file is add by huabin
#include "huabin.h"
/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef EULER_H
#define EULER_H

#include <cmath>

#include <gsl/gsl_math.h>

#include "Macro.h"
#include "Typedef.h"
#include "Random.h"
#include "Functions.h"

/**
 * Multiplication between two quaterions.
 *
 * @param dst result
 * @param a   left multiplier
 * @param b   right multiplier
 */
void quaternion_mul(dvec4& dst,
                    const dvec4& a,
                    const dvec4& b);

dvec4 quaternion_conj(const dvec4& quat);

/**
 * This function calculates phi and theta given a certain direction indicated by
 * a 3-vector.
 *
 * @param phi   phi
 * @param theta theta
 * @param src   3-vector indicating the direction
 */
void angle(double& phi,
           double& theta,
           const dvec3& src);

/**
 * This function calculates phi, theta and psi given the rotation matrix.
 *
 * @param phi   phi
 * @param theta theta
 * @param psi   psi
 * @param src   the rotation matrix
 */
void angle(double& phi,
           double& theta,
           double& psi,
           const dmat33& src);

/**
 * This function calculates phi, theta and psi given the quaternion indicated
 * by a 4-vector.
 *
 * @param phi   phi
 * @param theta theta
 * @param psi   psi
 * @param src   the quaternion
 */
void angle(double& phi,
           double& theta,
           double& psi,
           const dvec4& src);

/**
 * This function calculate the quaternion given phi, theta and psi.
 *
 * @param dst   the quaternion to be calculated
 * @param phi   phi
 * @param theta theta
 * @param psi   psi
 */
void quaternion(dvec4& dst,
                const double phi,
                const double theta,
                const double psi);

/**
 * This function calculates the quaternion given rotation angle and rotation axis.
 *
 * @param dst  the quaternion to be calculated
 * @param phi  the rotation angle
 * @param axis the rotation axis (unit vector)
 */
void quaternion(dvec4& dst,
                const double phi,
                const dvec3& axis);

void quaternion(dvec4& dst,
                const dmat33& src);

/**
 * This function calculates the rotation matrix given the a unit vector.
 *
 * @param dst the rotation matrix
 * @param vec the unit vector
 */
void rotate2D(dmat22& dst, const vec2& vec);

/**
 * This function calculates the rotation matrix given phi in 2D.
 *
 * @param dst the rotation matrix
 * @param phi phi
 */
void rotate2D(dmat22& dst, const double phi);

/**
 * This function calculates the direction vector given phi and theta. The 2-norm
 * of this direction vector is 1.
 *
 * @param dst   the direction vector
 * @param phi   phi
 * @param theta theta
 */
void direction(dvec3& dst,
               const double phi,
               const double theta);

/**
 * This function calculates the rotation matrix given phi, theta and psi.
 *
 * @param dst   the rotation matrix
 * @param phi   phi
 * @param theta theta
 * @param psi   psi
 */
void rotate3D(dmat33& dst,
              const double phi,
              const double theta,
              const double psi);

/**
 * This function calculates the rotation matrix given a quaternion.
 *
 * @param dst the rotation matrix
 * @param src the quaternion
 */
void rotate3D(dmat33& dst,
              const dvec4& src);

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

void randDirection(vec2& dir);

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
