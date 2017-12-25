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

int periodic(RFLOAT& x,
             const RFLOAT p)
{
    int n = floor(x / p);
    x -= n * p;
    return n;
}

void quaternion_mul(vec4& dst,
                    const vec4& a,
                    const vec4& b)
{
    RFLOAT w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    RFLOAT x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    RFLOAT y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    RFLOAT z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];

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
