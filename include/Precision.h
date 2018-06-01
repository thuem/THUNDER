#ifndef  HUABIN_H
#define  HUABIN_H

/*
 *#define TSGSL_cdf_chisq_Qinv gsl_cdf_chisq_Qinv
 *#define TSGSL_cdf_gaussian_Qinv gsl_cdf_gaussian_Qinv
 *#define TSGSL_complex_abs2 gsl_complex_abs2
 *#define TSGSL_fit_linear gsl_fit_linear
 *#define TSGSL_isinf gsl_isinf
 *#define TSGSL_isnan gsl_isnan
 *#define TSGSL_pow_2 gsl_pow_2
 *#define TSGSL_pow_3 gsl_pow_3
 *#define TSGSL_pow_4 gsl_pow_4
 *#define TSGSL_ran_bivariate_gaussian gsl_ran_bivariate_gaussian
 *#define TSGSL_ran_dir_2d gsl_ran_dir_2d
 *#define TSGSL_ran_flat gsl_ran_flat
 *#define TSGSL_ran_gaussian gsl_ran_gaussian
 *#define TSGSL_ran_shuffle gsl_ran_shuffle
 *#define TSGSL_rng_alloc gsl_rng_alloc
 *#define TSGSL_rng_free gsl_rng_free
 *#define TSGSL_rng_get gsl_rng_get
 *#define TSGSL_rng_set gsl_rng_set
 *#define TSGSL_rng_uniform gsl_rng_uniform
 *#define TSGSL_rng_uniform_int gsl_rng_uniform_int
 *#define TSGSL_sf_bessel_I0 gsl_sf_bessel_I0
 *#define TSGSL_sf_bessel_In gsl_sf_bessel_In
 *#define TSGSL_sf_bessel_Inu gsl_sf_bessel_Inu
 *#define TSGSL_sf_bessel_j0 gsl_sf_bessel_j0
 *#define TSGSL_sf_bessel_Jnu gsl_sf_bessel_Jnu
 *#define TSGSL_sf_sinc gsl_sf_sinc
 *#define TSGSL_sort gsl_sort
 *#define TSGSL_sort_largest gsl_sort_largest
 *#define TSGSL_stats_max gsl_stats_max
 *#define TSGSL_stats_mean gsl_stats_mean
 *#define TSGSL_stats_min gsl_stats_min
 *#define TSGSL_stats_quantile_from_sorted_data gsl_stats_quantile_from_sorted_data
 *#define TSGSL_stats_sd gsl_stats_sd
 *#define TSGSL_stats_sd_m gsl_stats_sd_m
 *
 */
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <immintrin.h>
#include "THUNDERConfig.h"
#include "Config.h"

/*
 *#define ENABLE_SIMD_256 1
 */

/*
 *#define ENABLE_SIMD_512 1
 */
#ifdef SINGLE_PRECISION
#include <fftw3.h>
#else
#include <fftw3.h>
#endif


#ifdef SINGLE_PRECISION
    typedef float RFLOAT;
    #define TSFFTW_COMPLEX fftwf_complex
    #define TSFFTW_PLAN fftwf_plan
    #define TS_MPI_DOUBLE MPI_FLOAT
    #define TS_MPI_DOUBLE_COMPLEX MPI_COMPLEX
    #define TS_MAX_RFLOAT_VALUE FLT_MAX
    typedef struct _complex_float_t
    {
        float dat[2];
    }Complex;
#else
    typedef double RFLOAT;
    #define TSFFTW_COMPLEX fftw_complex
    #define TSFFTW_PLAN fftw_plan
    #define TS_MPI_DOUBLE MPI_DOUBLE
    #define TS_MPI_DOUBLE_COMPLEX MPI_DOUBLE_COMPLEX
    #define TS_MAX_RFLOAT_VALUE DBL_MAX
    typedef struct _complex_float_t
    {
        double dat[2];
    }Complex;
#endif

inline RFLOAT TS_SIN(const RFLOAT x)
{
#ifdef SINGLE_PRECISION
    return sinf(x);
#else
    return sin(x);
#endif
};

inline RFLOAT TS_COS(const RFLOAT x)
{
#ifdef SINGLE_PRECISION
    return cosf(x);
#else
    return cos(x);
#endif
};

inline RFLOAT TS_SQRT(const RFLOAT x)
{
#ifdef SINGLE_PRECISION
    return sqrtf(x);
#else
    return sqrt(x);
#endif
};

inline RFLOAT TSGSL_MAX_RFLOAT(const RFLOAT a,
                               const RFLOAT b)
{
    return a > b ? a : b;
};

inline RFLOAT TSGSL_MIN_RFLOAT(const RFLOAT a,
                               const RFLOAT b)
{
    return a < b ? a : b;
};

RFLOAT TSGSL_cdf_chisq_Qinv (const RFLOAT Q, const RFLOAT nu);
RFLOAT TSGSL_cdf_gaussian_Qinv (const RFLOAT Q, const RFLOAT sigma);
RFLOAT TSGSL_complex_abs2 (Complex z);  /* return |z|^2 */
int TSGSL_isinf (const RFLOAT x);
int TSGSL_isnan (const RFLOAT x);
RFLOAT TSGSL_pow_2(const RFLOAT x);
RFLOAT TSGSL_pow_3(const RFLOAT x);
RFLOAT TSGSL_pow_4(const RFLOAT x);
void TSGSL_ran_bivariate_gaussian (const gsl_rng * r, RFLOAT sigma_x, RFLOAT sigma_y, RFLOAT rho, RFLOAT *x, RFLOAT *y);
int TSGSL_fit_linear (const RFLOAT * x, const size_t xstride, const RFLOAT * y, const size_t ystride, const size_t n, RFLOAT * c0, RFLOAT * c1, RFLOAT * cov00, RFLOAT * cov01, RFLOAT * cov11, RFLOAT * sumsq);
void TSGSL_ran_dir_2d (const gsl_rng * r, RFLOAT * x, RFLOAT * y);
RFLOAT TSGSL_ran_flat (const gsl_rng * r, const RFLOAT a, const RFLOAT b);
RFLOAT TSGSL_ran_gaussian (const gsl_rng * r, const RFLOAT sigma);
void TSGSL_ran_shuffle (const gsl_rng * r, void * base, size_t nmembm, size_t size);
gsl_rng *TSGSL_rng_alloc (const gsl_rng_type * T);
void TSGSL_rng_free (gsl_rng * r);
size_t TSGSL_rng_get (const gsl_rng * r);
void TSGSL_rng_set (const gsl_rng * r, size_t seed);
RFLOAT TSGSL_rng_uniform (const gsl_rng * r);
size_t TSGSL_rng_uniform_int (const gsl_rng * r, size_t n);
RFLOAT TSGSL_sf_bessel_I0(const RFLOAT x);
RFLOAT TSGSL_sf_bessel_In(const int n, const RFLOAT x);
RFLOAT TSGSL_sf_bessel_Inu(RFLOAT nu, RFLOAT x);
RFLOAT TSGSL_sf_bessel_j0(const RFLOAT x);
RFLOAT TSGSL_sf_bessel_Jnu(const RFLOAT nu, const RFLOAT x);
RFLOAT TSGSL_sf_sinc(const RFLOAT x);
void TSGSL_sort (RFLOAT * data, const size_t stride, const size_t n);
int TSGSL_sort_largest (RFLOAT * dst, const size_t k, const RFLOAT * src, const size_t stride, const size_t n);
void TSGSL_sort_index(size_t* dst, const RFLOAT* src, const size_t stride, const size_t n);
void TSGSL_sort_smallest_index(size_t* dst, const size_t k, const RFLOAT* src, const size_t stride, const size_t n);
void TSGSL_sort_largest_index(size_t* dst, const size_t k, const RFLOAT* src, const size_t stride, const size_t n);
RFLOAT TSGSL_stats_max (const RFLOAT data[], const size_t stride, const size_t n);
RFLOAT TSGSL_stats_mean (const RFLOAT data[], const size_t stride, const size_t n);
RFLOAT TSGSL_stats_min (const RFLOAT data[], const size_t stride, const size_t n);
RFLOAT TSGSL_stats_quantile_from_sorted_data (const RFLOAT sorted_data[], const size_t stride, const size_t n, const RFLOAT f) ;
RFLOAT TSGSL_stats_sd (const RFLOAT data[], const size_t stride, const size_t n);
RFLOAT TSGSL_stats_sd_m (const RFLOAT data[], const size_t stride, const size_t n, const RFLOAT mean);

int TSFFTW_init_threads(void);
void TSFFTW_cleanup_threads(void);
void TSFFTW_destroy_plan(TSFFTW_PLAN plan);
void TSFFTW_execute(const TSFFTW_PLAN plan);
void TSFFTW_execute_dft_r2c( const TSFFTW_PLAN p, RFLOAT *in, TSFFTW_COMPLEX *out);
void TSFFTW_execute_dft_c2r( const TSFFTW_PLAN p, TSFFTW_COMPLEX *in, RFLOAT *out); 
void *TSFFTW_malloc(size_t n);
void TSFFTW_free(void *p);

TSFFTW_PLAN TSFFTW_plan_dft_r2c_2d(int n0, int n1, RFLOAT *in, TSFFTW_COMPLEX *out, unsigned flags);
TSFFTW_PLAN TSFFTW_plan_dft_r2c_3d(int n0, int n1, int n2, RFLOAT *in, TSFFTW_COMPLEX *out, unsigned flags);

TSFFTW_PLAN TSFFTW_plan_dft_c2r_2d(int n0, int n1, TSFFTW_COMPLEX *in, RFLOAT *out, unsigned flags);
TSFFTW_PLAN TSFFTW_plan_dft_c2r_3d(int n0, int n1, int n2, TSFFTW_COMPLEX *in, RFLOAT *out, unsigned flags);

void TSFFTW_plan_with_nthreads(int nthreads);

void TSFFTW_set_timelimit(RFLOAT seconds);
#endif

