/*******************************************************************************
 * Author: Kongkun Yu, Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Functions.h"

int periodic(double& x,
             const double p)
{
    int n = floor(x / p);
    x -= n * p;
    return n;
}

void quaternion_mul(double* dst,
                    const double* a,
                    const double* b)
{
    double w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    double x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    double y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    double z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];

    dst[0] = w;
    dst[1] = x;
    dst[2] = y;
    dst[3] = z;
}

void normalise(gsl_vector& vec)
{
    double mean = gsl_stats_mean(vec.data, 1, vec.size);
    double stddev = gsl_stats_variance_m(vec.data, 1, vec.size, mean);

    gsl_vector_add_constant(&vec, -mean);
    gsl_vector_scale(&vec, 1.0 / stddev);
}

double MKB_FT(const double r,
              const double a,
              const double alpha)
{
    double u = r / a;

    if (u > 1)
        return 0;

    return (1 - gsl_pow_2(u))
         * gsl_sf_bessel_In(2, alpha * sqrt(1 - gsl_pow_2(u)))
         / gsl_sf_bessel_In(2, alpha);
}

double MKB_RL(const double r,
              const double a,
              const double alpha)
{
    double u = 2 * M_PI * a * r;

    double v = (u <= alpha) ? sqrt(gsl_pow_2(alpha) - gsl_pow_2(u))
                            : sqrt(gsl_pow_2(u) - gsl_pow_2(alpha));

    double w = pow(2 * M_PI, 1.5)
             * gsl_pow_3(a)
             * gsl_pow_2(alpha)
             / gsl_sf_bessel_In(2, alpha)
             / pow(v, 3.5);

    if (u <= alpha)
        return w * gsl_sf_bessel_Inu(3.5, v);
    else
        return w * gsl_sf_bessel_Jnu(3.5, v);
}

double TIK_RL(const double r)
{
    return gsl_pow_2(gsl_sf_bessel_j0(M_PI * r));
}
