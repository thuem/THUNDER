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

#include "Matrix.h"

void rotate2D(Matrix<double>& dst, const double phi);
// Return the rotating matrix for rotating for phi in 2D.
// This matrix should be on the left when calculating cooridinate.
// phi is radius, not degree.
// dst should be 2x2 matrix.

void normalVector(Vector<double>& dst,
                  const double phi,
                  const double theta);
// Return the direction of (phi, theta)
// dst should be a 3-vector.

void rotate3D(Matrix<double>& dst,
              const double phi,
              const double theta,
              const double psi);
// Return the rotating matrix for rotating Euler angle alpha, beta and gamma.
// dst should be a 3x3 matrix.

void rotate3DX(Matrix<double>& dst, const double phi);
void rotate3DY(Matrix<double>& dst, const double phi);
void rotate3DZ(Matrix<double>& dst, const double phi);

#endif // EULER_H 
