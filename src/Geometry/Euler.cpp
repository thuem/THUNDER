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

void rotate2D(mat22& dst, const double phi)
{
    double sine = sin(phi);
    double cosine = cos(phi);

    dst(0, 0) = cosine;
    dst(0, 1) = -sine;
    dst(1, 0) = sine;
    dst(1, 1) = cosine;
}

void direction(vec3& dst,
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

void angle(double& phi,
           double& theta,
           const vec3& src)
{
    theta = acos(src(2));
    phi = acos(src(0) / sin(theta));
    (src(1) / sin(theta) > 0) ? : (phi = 2 * M_PI - phi);
}

void rotate3D(mat33& dst,
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

void rotate3DX(mat33& dst, const double phi)
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

void rotate3DY(mat33& dst, const double phi)
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

void rotate3DZ(mat33& dst, const double phi)
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

void alignZ(mat33& dst,
            const vec3& vec)
{
    double x = vec(0);
    double y = vec(1);
    double z = vec(2);

    // compute the length of projection of YZ plane
    double pYZ = norm(vec.tail(2));
    // compute the length of this vector
    double p = norm(vec);

    if ((pYZ / p) > EQUAL_ACCURACY)
    {
        dst(0, 0) = pYZ;
        dst(0, 1) = -x * y / pYZ;
        dst(0, 2)  = -x * z / pYZ;

        dst(1, 0) = 0;
        dst(1, 1) = z / pYZ;
        dst(1, 2) = -y / pYZ;

        dst.row(2) = vec.t() / p;
    }
    else
    {
        dst.zeros();
        dst(0, 2) = -1;
        dst(1, 1) = 1;
        dst(2, 0) = 1;
    }
}

void rotate3D(mat33& dst,
              const double phi,
              const char axis)
{
    switch (axis)
    {
        case 'X':
            rotate3DX(dst, phi); break;
        
        case 'Y':
            rotate3DY(dst, phi); break;
        
        case 'Z':
            rotate3DZ(dst, phi); break;
    }
}

void rotate3D(mat33& dst,
              const double phi,
              const vec3& axis)
{
    mat33 A, R;

    alignZ(A, axis);

    rotate3DZ(R, phi);

    dst = A.t() * R * A;
}

void reflect3D(mat33& dst,
               const vec3& plane)
{
    mat33 A, M;

    alignZ(A, plane);

    M.eye();
    M(2, 2) = -1;

    dst = A.t() * M * A;
}

void translate3D(mat44& dst,
                 const vec3& vec)
{
    dst.eye();
    dst.col(3).head(3) = vec;
}

void scale3D(mat33& dst,
             const vec3& vec)
{
    dst.zeros();
    dst.diag() = vec;
}
