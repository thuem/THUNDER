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

#include <armadillo>
#include <gsl/gsl_math.h>

#include "Macro.h"
#include "Typedef.h"

using namespace arma;

void angle(double& phi,
           double& theta,
           const vec3& src);
/* Return phi and theta given the direction 
 * src should be a 3-vector */

void angle(double& phi,
           double& theta,
           double& psi,
           const mat33& src);

void angle(double& phi,
           double& theta,
           double& psi,
           const vec4& src);
// quaternion -> (phi, theta, psi)

void rotate2D(mat22& dst, const double phi);
// Return the rotating matrix for rotating for phi in 2D.
// This matrix should be on the left when calculating cooridinate.
// phi is radius, not degree.
// dst should be 2x2 matrix.

void direction(vec3& dst,
               const double phi,
               const double theta);
// Return the direction of (phi, theta)
// dst should be a 3-vector.

void rotate3D(mat33& dst,
              const double phi,
              const double theta,
              const double psi);
// Return the rotating matrix for rotating Euler angle alpha, beta and gamma.
// dst should be a 3x3 matrix.

void rotate3D(mat33& dst,
              const vec4& src);
// quaternion -> rotation matrix

void rotate3DX(mat33& dst, const double phi);
void rotate3DY(mat33& dst, const double phi);
void rotate3DZ(mat33& dst, const double phi);

void alignZ(mat33& dst,
            const vec3& vec);
// This function returns a 3x3 matrix.
// This matrix can align vec to Z axis.

void rotate3D(mat33& dst,
              const double phi,
              const char axis);
// This function returns a 3x3 matrix.
// X -> rotate around X axis
// Y -> rotate around Y axis
// Z -> rotate around Z axis

void rotate3D(mat33& dst,
              const double phi,
              const vec3& axis);
// This function returns a 3x3 matrix.
// This matrix represents a rotation around the axis given.

void reflect3D(mat33& dst,
               const vec3& plane);
// This function returns a 3x3 matrix.
// This matrix represents a reflection of the plane given.

void translate3D(mat44& dst,
                 const vec3& vec);
// This function returns a 4x4 matrix.
// This matrix represents a translation of the vec given.

void scale3D(mat33& dst,
             const vec3& vec);
// This function returns a 3x3 matrix.
// This matrix represents a scaling.
// scale along X -> vec[0]
// scale along Y -> vec[1]
// scale along Z -> vec[2]

#endif // EULER_H 
