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
           const vec3& src)
{
    theta = acos(src(2));
    phi = acos(src(0) / sin(theta));

    if (src(1) / sin(theta) <= 0)
        (phi = 2 * M_PI - phi);
}

void angle(double& phi,
           double& theta,
           double& psi,
           const mat33& src)
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
           const vec4& src)
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

void quaternion(vec4& dst,
                const double phi,
                const double theta,
                const double psi)
{
    dst(0) = cos((phi + psi) / 2) * cos(theta / 2);
    dst(1) = cos((phi - psi) / 2) * sin(theta / 2);
    dst(2) = sin((phi - psi) / 2) * sin(theta / 2);
    dst(3) = sin((phi + psi) / 2) * cos(theta / 2);
}

void quaternion(vec4& dst,
                const double phi,
                const vec3& axis)
{
    dst(0) = cos(phi / 2);
    dst(1) = sin(phi / 2) * axis(0);
    dst(2) = sin(phi / 2) * axis(1);
    dst(3) = sin(phi / 2) * axis(2);
}

void quaternion(vec4& dst,
                const mat33& src)
{
    dst(0) = 0.5 * sqrt(GSL_MAX_DBL(0, 1 + src(0, 0) + src(1, 1) + src(2, 2)));
    dst(1) = 0.5 * sqrt(GSL_MAX_DBL(0, 1 + src(0, 0) - src(1, 1) - src(2, 2)));
    dst(2) = 0.5 * sqrt(GSL_MAX_DBL(0, 1 - src(0, 0) + src(1, 1) - src(2, 2)));
    dst(3) = 0.5 * sqrt(GSL_MAX_DBL(0, 1 - src(0, 0) - src(1, 1) + src(2, 2)));

    dst(1) = copysign(dst(1), src(2, 1) - src(1, 2));
    dst(2) = copysign(dst(2), src(0, 2) - src(2, 0));
    dst(3) = copysign(dst(3), src(1, 0) - src(0, 1));

    /***
    if (src.trace() > 0)
    {
        double s = sqrt(src.trace() + 1);

        dst(0) = 0.5 * s;
        dst(1) = 0.5 * (src(2, 1) - src(1, 2)) / s;
        dst(2) = 0.5 * (src(0, 2) - src(2, 0)) / s;
        dst(3) = 0.5 * (src(1, 0) - src(0, 1)) / s;
    }
    else
    {
        if ((src(0, 0) > src(1, 1)) &&
            (src(0, 0) > src(2, 2)))
        {
            // src(0, 0) -> biggest
            double s = sqrt(src(0, 0) - src(1, 1) - src(2, 2) + 1);

            dst(0) = 0.5 * (dst(2, 1) - dst(1, 2)) / s;
            dst(1) = 0.5 * s;
            dst(2) = 0.5 * (src(0, 1) + src(1, 0)) / s;
            dst(3) = 0.5 * (src(0, 2) + src(2, 0)) / s;
        }
        else if (src(1, 1) > src(2, 2))
        {
            // src(1, 1) -> biggest

            double s = sqrt(src(1, 1) - src(0, 0) - src(2, 2) + 1);

            dst(0) = 0.5 * (dst(0, 2) - dst(2, 0)) / s;
            dst(1) = 0.5 * (src(0, 1) + src(1, 0)) / s;
            dst(2) = 0.5 * s;
            dst(3) = 0.5 * (src(1, 2) + src(2, 1)) / s;
        }
        else
        {
            // src(2, 2) -> biggest

            double s = sqrt(src(2, 2) - src(0, 0) - src(1, 1) + 1);

            dst(0) = 0.5 * (dst(1, 0) - dst(0, 1)) / s;
            dst(1) = 0.5 * (src(0, 2) + src(2, 0)) / s;
            dst(2) = 0.5 * (src(1, 2) + src(2, 1)) / s;
            dst(3) = 0.5 * s;
        }
    }
    ***/
}

void rotate2D(mat22& dst, const vec2& vec)
{
    /***
    vec(0) = cos(phi);
    vec(1) = sin(phi);
    ***/

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

void rotate3D(mat33& dst,
              const vec4& src)
{
    /***
    mat33 A = {{0, -src(3), src(2)},
               {src(3), 0, -src(1)},
               {-src(2), src(1), 0}};
    dst = mat33(fill::eye) + 2 * src(0) * A + 2 * A * A;
    ***/

    mat33 A;
    A << 0, -src(3), src(2),
         src(3), 0, -src(1),
         -src(2), src(1), 0;
    dst = mat33::Identity() + 2 * src(0) * A + 2 * A * A;
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
    double pYZ = vec.tail<2>().norm();
    // double pYZ = norm(vec.tail(2));
    // compute the length of this vector
    double p = vec.norm();
    // double p = norm(vec);

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
    vec4 quat;

    quaternion(quat, phi, axis);

    rotate3D(dst, quat);

    /***
    mat33 A, R;

    alignZ(A, axis);

    rotate3DZ(R, phi);

    dst = A.transpose() * R * A;
    ***/
}

void reflect3D(mat33& dst,
               const vec3& plane)
{
    mat33 A, M;

    alignZ(A, plane);

    M.setIdentity();
    M(2, 2) = -1;

    dst = A.transpose() * M * A;
}

void translate3D(mat44& dst,
                 const vec3& vec)
{
    dst.setIdentity();
    dst.col(3).head<3>() = vec;
}

void scale3D(mat33& dst,
             const vec3& vec)
{
    dst.setZero();
    dst(0, 0) = vec(0);
    dst(1, 1) = vec(1);
    dst(2, 2) = vec(2);
}

void swingTwist(vec4& swing,
                vec4& twist,
                const vec4& src,
                const vec3& vec)
{
    double p = vec3(src(1), src(2), src(3)).dot(vec);

    twist = vec4(src(0), p * vec(0), p * vec(1), p * vec(2));

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

void randQuaternion(vec4& quat)
{
    gsl_rng* engine = get_random_engine();

    quat(0) = gsl_ran_gaussian(engine, 1);
    quat(1) = gsl_ran_gaussian(engine, 1);
    quat(2) = gsl_ran_gaussian(engine, 1);
    quat(3) = gsl_ran_gaussian(engine, 1);

    quat /= quat.norm();
}

void randRotate3D(mat33& rot)
{
    vec4 quat;

    randQuaternion(quat);

    rotate3D(rot, quat);
}
