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

void homogenize(Matrix<double>& mat)
{
    Matrix<double> temp = mat;

    mat.resize(4, 4);
    mat.zeros();
    
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            mat.set(temp.get(i, j), i, j);

    mat.set(1, 3, 3);
}

void alignZ(Matrix<double>& dst,
            const Vector<double>& vec)
{
    double x = vec.get(0);
    double y = vec.get(1);
    double z = vec.get(2);

    // compute the length of projection of YZ plane
    double pYZ = sqrt(pow(y, 2) + pow(z, 2));
    // compute the length of this vector
    double p = vec.modulus();

    if ((pYZ / p) > EQUAL_ACCURACY)
    {
        dst(0, 0) = pYZ;
        dst(0, 1) = -x * y / pYZ;
        dst(0, 2)  = -x * z / pYZ;

        dst(1, 0) = 0;
        dst(1, 1) = z / pYZ;
        dst(1, 2) = -y / pYZ;

        dst(2, 0) = x / p;
        dst(2, 1) = y / p;
        dst(2, 2) = z / p;
    }
    else
    {
        dst(0, 0) = 0;
        dst(0, 1) = 0;
        dst(0, 2) = -1;

        dst(1, 0) = 0;
        dst(1, 1) = 1;
        dst(1, 2) = 0;

        dst(2, 0) = 1;
        dst(2, 1) = 0;
        dst(2, 2) = 0;
    }
}

void rotate3D(Matrix<double>& dst,
              const double phi,
              const char axis)
{
    switch (axis)
    {
        case 'X':
            rotate3DX(dst, phi);
        
        case 'Y':
            rotate3DY(dst, phi);
        
        case 'Z':
            rotate3DZ(dst, phi);
    }
}

void rotate3D(Matrix<double>& dst,
              const double phi,
              const Vector<double>& axis)
{
    Matrix<double> A(3, 3);
    alignZ(A, axis);

    Matrix<double> R(3, 3);
    rotate3DZ(R, phi);

    dst = A.transpose() * R * A;
}

void reflect3D(Matrix<double>& dst,
               const Vector<double>& plane)
{
    Matrix<double> A(3, 3);
    alignZ(A, plane);

    Matrix<double> M(3, 3);
    M.zeros();
    M(2, 2) = -1;

    dst = A.transpose() * M * A;
}

void translate3D(Matrix<double>& dst,
                 const Vector<double>& vec)
{
    dst.identity();

    dst.set(vec.get(0), 0, 3);
    dst.set(vec.get(1), 1, 3);
    dst.set(vec.get(2), 2, 3);

    /*********************
     * [ 1  0  0  vec[0] ]
     * [ 0  1  0  vec[1] ]
     * [ 0  0  1  vec[2] ]
     * [ 0  0  0   1     ]
     * ******************/
}

void scale3D(Matrix<double>& dst,
             const Vector<double>& vec)
{
    dst.zeros();

    for (int i = 0; i < 3; i++)
        dst.set(vec.get(i), i, i);
}
