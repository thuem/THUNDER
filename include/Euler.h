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

using namespace arma;

void rotate2D(mat22& dst, const double phi);
// Return the rotating matrix for rotating for phi in 2D.
// This matrix should be on the left when calculating cooridinate.
// phi is radius, not degree.
// dst should be 2x2 matrix.

void normalVector(vec3& dst,
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

void rotate3DX(mat33& dst, const double phi);
void rotate3DY(mat33& dst, const double phi);
void rotate3DZ(mat33& dst, const double phi);

#endif // EULER_H 
