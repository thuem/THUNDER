//This header file is add by huabin
#include "huabin.h"
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

    RFLOAT s = 0;

    for (size_t i = 0; i < (size_t)v.size(); i++)
    {
        s += v(i);

        sum(i) = s;
    }

    return sum;
}

dvec d_cumsum(const dvec& v)
{
    dvec sum(v.size());

    double s = 0;

    for (size_t i = 0; i < (size_t)v.size(); i++)
    {
        s += v(i);

        sum(i) = s;
    }

    return sum;
}

uvec index_sort_ascend(const vec& v)
{
    uvec idx(v.size());

    TSGSL_sort_smallest_index(idx.data(), idx.size(), v.data(), 1, v.size());

    return idx;
}

uvec d_index_sort_ascend(const dvec& v)
{
    uvec idx(v.size());

    gsl_sort_smallest_index(idx.data(), idx.size(), v.data(), 1, v.size());

    return idx;
}

uvec index_sort_descend(const vec& v)
{
    uvec idx(v.size());

    TSGSL_sort_largest_index(idx.data(), idx.size(), v.data(), 1, v.size());

    return idx;
}

uvec d_index_sort_descend(const dvec& v)
{
    uvec idx(v.size());

    gsl_sort_largest_index(idx.data(), idx.size(), v.data(), 1, v.size());

    return idx;
}

int periodic(RFLOAT& x,
             const RFLOAT p)
{
    int n = floor(x / p);
    x -= n * p;
    return n;
}
RFLOAT MKB_FT(const RFLOAT r,
              const RFLOAT a,
              const RFLOAT alpha)
{
    RFLOAT u = r / a;

    if (u > 1) return 0;

#ifdef FUNCTIONS_MKB_ORDER_2
    return (1 - TSGSL_pow_2(u))
         * TSGSL_sf_bessel_In(2, alpha * sqrt(1 - TSGSL_pow_2(u)))
         / TSGSL_sf_bessel_In(2, alpha);
#endif

#ifdef FUNCTIONS_MKB_ORDER_0
    return TSGSL_sf_bessel_I0(alpha * sqrt(1 - TSGSL_pow_2(u)))
         / TSGSL_sf_bessel_I0(alpha);
#endif
}

RFLOAT MKB_FT_R2(const RFLOAT r2,
                 const RFLOAT a,
                 const RFLOAT alpha)
{
    RFLOAT u2 = r2 / TSGSL_pow_2(a);

    if (u2 > 1) return 0;

#ifdef FUNCTIONS_MKB_ORDER_2
    return (1 - u2)
         * TSGSL_sf_bessel_In(2, alpha * sqrt(1 - u2))
         / TSGSL_sf_bessel_In(2, alpha);
#endif

#ifdef FUNCTIONS_MKB_ORDER_0
    return TSGSL_sf_bessel_I0(alpha * sqrt(1 - u2))
         / TSGSL_sf_bessel_I0(alpha);
#endif
}

RFLOAT MKB_RL(const RFLOAT r,
              const RFLOAT a,
              const RFLOAT alpha)
{
    RFLOAT u = 2 * M_PI * a * r;

    RFLOAT v = (u <= alpha) ? sqrt(TSGSL_pow_2(alpha) - TSGSL_pow_2(u))
                            : sqrt(TSGSL_pow_2(u) - TSGSL_pow_2(alpha));

#ifdef FUNCTIONS_MKB_ORDER_2
    RFLOAT w = pow(2 * M_PI, 1.5)
             * TSGSL_pow_3(a)
             * TSGSL_pow_2(alpha)
             / TSGSL_sf_bessel_In(2, alpha)
             / pow(v, 3.5);

    if (u <= alpha)
        return w * TSGSL_sf_bessel_Inu(3.5, v);
    else
        return w * TSGSL_sf_bessel_Jnu(3.5, v);
#endif

#ifdef FUNCTIONS_MKB_ORDER_0
    RFLOAT w = pow(2 * M_PI, 1.5)
             * TSGSL_pow_3(a)
             / TSGSL_sf_bessel_I0(alpha)
             / pow(v, 1.5);

    if (u <= alpha)
        return w * TSGSL_sf_bessel_Inu(1.5, v);
    else
        return w * TSGSL_sf_bessel_Jnu(1.5, v);
#endif
}

RFLOAT MKB_RL_R2(const RFLOAT r2,
                 const RFLOAT a,
                 const RFLOAT alpha)
{
    RFLOAT u2 = TSGSL_pow_2(2 * M_PI * a) * r2;

    RFLOAT v = (u2 <= TSGSL_pow_2(alpha))
             ? sqrt(TSGSL_pow_2(alpha) - u2)
             : sqrt(u2 - TSGSL_pow_2(alpha));

#ifdef FUNCTIONS_MKB_ORDER_2
    RFLOAT w = pow(2 * M_PI, 1.5)
             * TSGSL_pow_3(a)
             * TSGSL_pow_2(alpha)
             / TSGSL_sf_bessel_In(2, alpha)
             / pow(v, 3.5);

    if (u2 <= TSGSL_pow_2(alpha))
        return w * TSGSL_sf_bessel_Inu(3.5, v);
    else
        return w * TSGSL_sf_bessel_Jnu(3.5, v);
#endif

#ifdef FUNCTIONS_MKB_ORDER_0
    RFLOAT w = pow(2 * M_PI, 1.5)
             * TSGSL_pow_3(a)
             / TSGSL_sf_bessel_I0(alpha)
             / pow(v, 1.5);

    if (u2 <= TSGSL_pow_2(alpha))
        return w * TSGSL_sf_bessel_Inu(1.5, v);
    else
        return w * TSGSL_sf_bessel_Jnu(1.5, v);
#endif
}

RFLOAT MKB_BLOB_VOL(const RFLOAT a,
                    const RFLOAT alpha)
{
#ifdef FUNCTIONS_MKB_ORDER_2
    return pow(2 * M_PI / alpha, 1.5)
         * TSGSL_sf_bessel_Inu(3.5, alpha)
         / TSGSL_sf_bessel_In(2, alpha)
         * TSGSL_pow_3(a);
#endif

#ifdef FUNCTIONS_MKB_ORDER_0
    return pow(2 * M_PI / alpha, 1.5)
         * TSGSL_sf_bessel_Inu(1.5, alpha)
         / TSGSL_sf_bessel_I0(alpha)
         * TSGSL_pow_3(a);
#endif
}

RFLOAT TIK_RL(const RFLOAT r)
{
    return TSGSL_pow_2(TSGSL_sf_bessel_j0(M_PI * r));
}

RFLOAT NIK_RL(const RFLOAT r)
{
    return TSGSL_sf_bessel_j0(M_PI * r);
}

RFLOAT median(vec src,
              const int n)
{
    TSGSL_sort(src.data(), 1, n);

    return TSGSL_stats_quantile_from_sorted_data(src.data(), 1, n, 0.5);
}

void stat_MAS(RFLOAT& mean,
              RFLOAT& std,
              vec src,
              const int n)
{
    TSGSL_sort(src.data(), 1, n);

    mean = TSGSL_stats_quantile_from_sorted_data(src.data(),
                                                 1,
                                                 n,
                                                 0.5);

    src = abs(src.array() - mean);

    TSGSL_sort(src.data(), 1, n);

    std = TSGSL_stats_quantile_from_sorted_data(src.data(),
                                                1,
                                                n,
                                                0.5)
        * 1.4826;
}
