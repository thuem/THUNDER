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

vec cumsum(const vec& v)
{
    vec sum(v.size());

    std::partial_sum(v.data(), v.data() + v.size(), sum.data());

    return sum;
}

struct IndexSortAscendComparator
{
    const vec* pv;
    bool operator()(unsigned int i, unsigned int j) const
    {
        return (*pv)(i) < (*pv)(j);
    }
};

uvec index_sort_ascend(const vec& v)
{
    uvec idx(v.size());

    for (unsigned int i = 0; i < idx.size(); i++)
        idx(i) = i;

    IndexSortAscendComparator cmp;
    cmp.pv = &v;
    sort(idx.data(), idx.data() + idx.size(), cmp);

    return idx;
}

struct IndexSortDescendComparator
{
    const vec* pv;
    bool operator()(unsigned int i, unsigned int j) const
    {
        return (*pv)(i) > (*pv)(j);
    }
};

uvec index_sort_descend(const vec& v)
{
    uvec idx(v.size());

    for (unsigned int i = 0; i < idx.size(); i++)
        idx(i) = i;

    IndexSortDescendComparator cmp;
    cmp.pv = &v;
    sort(idx.data(), idx.data() + idx.size(), cmp);

    return idx;
}

int periodic(double& x,
             const double p)
{
    int n = floor(x / p);
    x -= n * p;
    return n;
}

void quaternion_mul(vec4& dst,
                    const vec4& a,
                    const vec4& b)
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

vec4 quaternion_conj(const vec4& quat)
{
    vec4 conj;

    conj << quat(0),
            -quat(1),
            -quat(2),
            -quat(3);

    return conj;
}

double MKB_FT(const double r,
              const double a,
              const double alpha)
{
    double u = r / a;

    if (u > 1) return 0;

#ifdef FUNCTIONS_MKB_ORDER_2
    return (1 - gsl_pow_2(u))
         * gsl_sf_bessel_In(2, alpha * sqrt(1 - gsl_pow_2(u)))
         / gsl_sf_bessel_In(2, alpha);
#endif

#ifdef FUNCTIONS_MKB_ORDER_0
    return gsl_sf_bessel_I0(alpha * sqrt(1 - gsl_pow_2(u)))
         / gsl_sf_bessel_I0(alpha);
#endif
}

double MKB_FT_R2(const double r2,
                 const double a,
                 const double alpha)
{
    double u2 = r2 / gsl_pow_2(a);

    if (u2 > 1) return 0;

#ifdef FUNCTIONS_MKB_ORDER_2
    return (1 - u2)
         * gsl_sf_bessel_In(2, alpha * sqrt(1 - u2))
         / gsl_sf_bessel_In(2, alpha);
#endif

#ifdef FUNCTIONS_MKB_ORDER_0
    return gsl_sf_bessel_I0(alpha * sqrt(1 - u2))
         / gsl_sf_bessel_I0(alpha);
#endif
}

double MKB_RL(const double r,
              const double a,
              const double alpha)
{
    double u = 2 * M_PI * a * r;

    double v = (u <= alpha) ? sqrt(gsl_pow_2(alpha) - gsl_pow_2(u))
                            : sqrt(gsl_pow_2(u) - gsl_pow_2(alpha));

#ifdef FUNCTIONS_MKB_ORDER_2
    double w = pow(2 * M_PI, 1.5)
             * gsl_pow_3(a)
             * gsl_pow_2(alpha)
             / gsl_sf_bessel_In(2, alpha)
             / pow(v, 3.5);

    if (u <= alpha)
        return w * gsl_sf_bessel_Inu(3.5, v);
    else
        return w * gsl_sf_bessel_Jnu(3.5, v);
#endif

#ifdef FUNCTIONS_MKB_ORDER_0
    double w = pow(2 * M_PI, 1.5)
             * gsl_pow_3(a)
             / gsl_sf_bessel_I0(alpha)
             / pow(v, 1.5);

    if (u <= alpha)
        return w * gsl_sf_bessel_Inu(1.5, v);
    else
        return w * gsl_sf_bessel_Jnu(1.5, v);
#endif
}

double MKB_RL_R2(const double r2,
                 const double a,
                 const double alpha)
{
    double u2 = gsl_pow_2(2 * M_PI * a) * r2;

    double v = (u2 <= gsl_pow_2(alpha))
             ? sqrt(gsl_pow_2(alpha) - u2)
             : sqrt(u2 - gsl_pow_2(alpha));

#ifdef FUNCTIONS_MKB_ORDER_2
    double w = pow(2 * M_PI, 1.5)
             * gsl_pow_3(a)
             * gsl_pow_2(alpha)
             / gsl_sf_bessel_In(2, alpha)
             / pow(v, 3.5);

    if (u2 <= gsl_pow_2(alpha))
        return w * gsl_sf_bessel_Inu(3.5, v);
    else
        return w * gsl_sf_bessel_Jnu(3.5, v);
#endif

#ifdef FUNCTIONS_MKB_ORDER_0
    double w = pow(2 * M_PI, 1.5)
             * gsl_pow_3(a)
             / gsl_sf_bessel_I0(alpha)
             / pow(v, 1.5);

    if (u2 <= gsl_pow_2(alpha))
        return w * gsl_sf_bessel_Inu(1.5, v);
    else
        return w * gsl_sf_bessel_Jnu(1.5, v);
#endif
}

double MKB_BLOB_VOL(const double a,
                    const double alpha)
{
#ifdef FUNCTIONS_MKB_ORDER_2
    return pow(2 * M_PI / alpha, 1.5)
         * gsl_sf_bessel_Inu(3.5, alpha)
         / gsl_sf_bessel_In(2, alpha)
         * gsl_pow_3(a);
#endif

#ifdef FUNCTIONS_MKB_ORDER_0
    return pow(2 * M_PI / alpha, 1.5)
         * gsl_sf_bessel_Inu(1.5, alpha)
         / gsl_sf_bessel_I0(alpha)
         * gsl_pow_3(a);
#endif
}

double TIK_RL(const double r)
{
    return gsl_pow_2(gsl_sf_bessel_j0(M_PI * r));
}

double NIK_RL(const double r)
{
    return gsl_sf_bessel_j0(M_PI * r);
}

double median(vec src,
              const int n)
{
    gsl_sort(src.data(), 1, n);

    return gsl_stats_quantile_from_sorted_data(src.data(), 1, n, 0.5);
}

void stat_MAS(double& mean,
              double& std,
              vec src,
              const int n)
{
    gsl_sort(src.data(), 1, n);

    mean = gsl_stats_quantile_from_sorted_data(src.data(),
                                               1,
                                               n,
                                               0.5);

    src = abs(src.array() - mean);

    gsl_sort(src.data(), 1, n);

    std = gsl_stats_quantile_from_sorted_data(src.data(),
                                              1,
                                              n,
                                              0.5)
        * 1.4826;
}
