/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Functions.h"

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
