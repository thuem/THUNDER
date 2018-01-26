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

#include "Euler.h"

void angle(double& phi,
           double& theta,
           const dvec3& src)
{
    theta = acos(src(2));
    phi = acos(src(0) / sin(theta));

    if (src(1) / sin(theta) <= 0)
        (phi = 2 * M_PI - phi);
}

void angle(double& phi,
           double& theta,
           double& psi,
           const dmat33& src)
{
    theta = acos(src(2, 2));
    psi = acos(src(2, 1) / sin(theta));

    if (src(2, 0) / sin(theta) <= 0)
        (psi = 2 * M_PI - psi);

    phi = acos(-src(1, 2) / sin(theta));

    if (src(0, 2) / sin(theta) <= 0)
        (phi = 2 * M_PI - phi);
}

void angle(double& phi,
           double& theta,
           double& psi,
           const dvec4& src)
{
    phi = atan2((src(1) * src(3) + src(0) * src(2)),
                (src(0) * src(1) - src(2) * src(3)));

    if (phi < 0) phi += 2 * M_PI;

    theta = acos(gsl_pow_2(src(0))
               - gsl_pow_2(src(1))
               - gsl_pow_2(src(2))
               + gsl_pow_2(src(3)));

    psi = atan2((src(1) * src(3) - src(0) * src(2)),
                (src(0) * src(1) + src(2) * src(3)));

    if (psi < 0) psi += 2 * M_PI;
}

void quaternion(dvec4& dst,
                const double phi,
                const double theta,
                const double psi)
{
    dst(0) = cos((phi + psi) / 2) * cos(theta / 2);
    dst(1) = cos((phi - psi) / 2) * sin(theta / 2);
    dst(2) = sin((phi - psi) / 2) * sin(theta / 2);
    dst(3) = sin((phi + psi) / 2) * cos(theta / 2);
}

void quaternion(dvec4& dst,
                const double phi,
                const dvec3& axis)
{
    dst(0) = cos(phi / 2);
    dst(1) = sin(phi / 2) * axis(0);
    dst(2) = sin(phi / 2) * axis(1);
    dst(3) = sin(phi / 2) * axis(2);
}

void quaternion(dvec4& dst,
                const dmat33& src)
{
    dst(0) = 0.5 * sqrt(GSL_MAX_DBL(0, 1 + src(0, 0) + src(1, 1) + src(2, 2)));
    dst(1) = 0.5 * sqrt(GSL_MAX_DBL(0, 1 + src(0, 0) - src(1, 1) - src(2, 2)));
    dst(2) = 0.5 * sqrt(GSL_MAX_DBL(0, 1 - src(0, 0) + src(1, 1) - src(2, 2)));
    dst(3) = 0.5 * sqrt(GSL_MAX_DBL(0, 1 - src(0, 0) - src(1, 1) + src(2, 2)));

    dst(1) = copysign(dst(1), src(2, 1) - src(1, 2));
    dst(2) = copysign(dst(2), src(0, 2) - src(2, 0));
    dst(3) = copysign(dst(3), src(1, 0) - src(0, 1));
}

void rotate2D(mat22& dst, const vec2& vec)
{
    dst(0, 0) = vec(0);
    dst(0, 1) = -vec(1);
    dst(1, 0) = vec(1);
    dst(1, 1) = vec(0);
}

void rotate2D(mat22& dst, const double phi)
{
    double sine = sin(phi);
    double cosine = cos(phi);

    dst(0, 0) = cosine;
    dst(0, 1) = -sine;
    dst(1, 0) = sine;
    dst(1, 1) = cosine;
}

void direction(dvec3& dst,
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

void rotate3D(dmat33& dst,
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

void rotate3D(dmat33& dst,
              const dvec4& src)
{
    dmat33 A;
    A << 0, -src(3), src(2),
         src(3), 0, -src(1),
         -src(2), src(1), 0;
    dst = dmat33::Identity() + 2 * src(0) * A + 2 * A * A;
}

void rotate3DX(dmat33& dst, const double phi)
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

void rotate3DY(dmat33& dst, const double phi)
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

void rotate3DZ(dmat33& dst, const double phi)
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

void alignZ(dmat33& dst,
            const dvec3& vec)
{
    double x = vec(0);
    double y = vec(1);
    double z = vec(2);

    // compute the length of projection of YZ plane
    double pYZ = vec.tail<2>().norm();

    double p = vec.norm();

    if ((pYZ / p) > EQUAL_ACCURACY)
    {
        dst(0, 0) = pYZ;
        dst(0, 1) = -x * y / pYZ;
        dst(0, 2)  = -x * z / pYZ;

        dst(1, 0) = 0;
        dst(1, 1) = z / pYZ;
        dst(1, 2) = -y / pYZ;

        dst.row(2) = vec.transpose() / p;
    }
    else
    {
        dst.setZero();
        dst(0, 2) = -1;
        dst(1, 1) = 1;
        dst(2, 0) = 1;
    }
}

void rotate3D(dmat33& dst,
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

void rotate3D(dmat33& dst,
              const double phi,
              const dvec3& axis)
{
    dvec4 quat;

    quaternion(quat, phi, axis);

    rotate3D(dst, quat);
}

void reflect3D(dmat33& dst,
               const dvec3& plane)
{
    dmat33 A, M;

    alignZ(A, plane);

    M.setIdentity();
    M(2, 2) = -1;

    dst = A.transpose() * M * A;
}

void translate3D(dmat44& dst,
                 const dvec3& vec)
{
    dst.setIdentity();
    dst.col(3).head<3>() = vec;
}

void scale3D(dmat33& dst,
             const dvec3& vec)
{
    dst.setZero();
    dst(0, 0) = vec(0);
    dst(1, 1) = vec(1);
    dst(2, 2) = vec(2);
}

void swingTwist(dvec4& swing,
                dvec4& twist,
                const dvec4& src,
                const dvec3& vec)
{
    double p = dvec3(src(1), src(2), src(3)).dot(vec);

    twist = dvec4(src(0), p * vec(0), p * vec(1), p * vec(2));

    twist /= twist.norm();

    quaternion_mul(swing, src, quaternion_conj(twist));
}

void randDirection(vec2& dir)
{
    gsl_rng* engine = get_random_engine();

    dir(0) = gsl_ran_gaussian(engine, 1);
    dir(1) = gsl_ran_gaussian(engine, 1);

    dir /= dir.norm();
}

void randRotate2D(mat22& rot)
{
    vec2 dir;
    randDirection(dir);
    
    rotate2D(rot, dir);
}

void randQuaternion(dvec4& quat)
{
    gsl_rng* engine = get_random_engine();

    quat(0) = gsl_ran_gaussian(engine, 1);
    quat(1) = gsl_ran_gaussian(engine, 1);
    quat(2) = gsl_ran_gaussian(engine, 1);
    quat(3) = gsl_ran_gaussian(engine, 1);

    quat /= quat.norm();
}

void randRotate3D(dmat33& rot)
{
    dvec4 quat;

    randQuaternion(quat);

    rotate3D(rot, quat);
}
