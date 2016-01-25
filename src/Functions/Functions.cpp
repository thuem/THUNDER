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
