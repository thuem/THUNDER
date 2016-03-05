/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Transformation.h"

void alignZ(mat33& dst,
            const vec3& vec)
{
    double x = vec(0);
    double y = vec(1);
    double z = vec(2);

    // compute the length of projection of YZ plane
    double pYZ = norm(vec.head(2));
    // double pYZ = sqrt(pow(y, 2) + pow(z, 2));
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

        dst.row(2) = vec / p;
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
