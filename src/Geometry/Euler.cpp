/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Euler.h"

void rotate2D(Matrix<double>& dst, const double phi)
{
    double sine = sin(phi);
    double cosine = cos(phi);

    dst(0, 0) = cosine;
    dst(0, 1) = -sine;
    dst(1, 0) = sine;
    dst(1, 1) = cosine;
}

void normalVector(Vector<double>& dst,
                  const double phi,
                  const double theta)
{
    double sinPhi = sin(phi);
    double cosPhi = cos(phi);
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);

    dst(0) = sinTheta * cosPhi;
    dst(1) = sinTheta * sinPhi;
    dst(2) = cosTheta;
}

void rotate3D(Matrix<double>& dst,
              const double phi,
              const double theta,
              const double psi)
{ 
    double sinPhi = sin(phi);
    double cosPhi = cos(phi);
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);
    double sinPsi = sin(psi);
    double cosPsi = cos(psi);

    dst(0, 0) = cosPhi * cosPsi - sinPhi * cosTheta * sinPsi;
    dst(0, 1) = -cosPhi * sinPsi - sinPhi * cosTheta * cosPsi;
    dst(0, 2) = sinPhi * sinTheta;
    dst(1, 0) = sinPhi * cosPsi + cosPhi * cosTheta * sinPsi;
    dst(1, 1) = -sinPhi * sinPsi + cosPhi * cosTheta * cosPsi;
    dst(1, 2) = -cosPhi * sinTheta;
    dst(2, 0) = sinTheta * sinPsi;
    dst(2, 1) = sinTheta * cosPsi;
    dst(2, 2) = cosTheta;
} 

void rotate3DX(Matrix<double>& dst, const double phi)
{
    double sine = sin(phi);
    double cosine = cos(phi);

    dst(0, 0) = 1;
    dst(0, 1) = 0;
    dst(0, 2) = 0;
    dst(1, 0) = 0;
    dst(1, 1) = cosine;
    dst(1, 2) = -sine;
    dst(2, 0) = 0;
    dst(2, 1) = sine;
    dst(2, 2) = cosine;
}

void rotate3DY(Matrix<double>& dst, const double phi)
{
    double sine = sin(phi);
    double cosine = cos(phi);

    dst(0, 0) = cosine;
    dst(0, 1) = 0;
    dst(0, 2) = sine;
    dst(1, 0) = 0;
    dst(1, 1) = 1; 
    dst(1, 2) = 0;
    dst(2, 0) = -sine; 
    dst(2, 1) = 0; 
    dst(2, 2) = cosine;
}

void rotate3DZ(Matrix<double>& dst, const double phi)
{
    double sine = sin(phi);
    double cosine = cos(phi);

    dst(0, 0) = cosine;
    dst(0, 1) = -sine; 
    dst(0, 2) = 0; 
    dst(1, 0) = sine; 
    dst(1, 1) = cosine; 
    dst(1, 2) = 0;
    dst(2, 0) = 0; 
    dst(2, 1) = 0; 
    dst(2, 2) = 1; 
}
