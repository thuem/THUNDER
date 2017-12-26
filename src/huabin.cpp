/*******************************************************************
 *       Filename:  huabin.cpp                                     
 *                                                                 
 *    Description:                                        
 *                                                                 
 *        Version:  1.0                                            
 *        Created:  12/25/2017 02:37:45 PM                                 
 *       Revision:  none                                           
 *       Compiler:  gcc                                           
 *                                                                 
 *         Author:  Ruan Huabin                                      
 *          Email:  ruanhuabin@tsinghua.edu.cn                                        
 *        Company:  Dep. of CS, Tsinghua Unversity                                      
 *                                                                 
 *******************************************************************/
#include "huabin.h"

RFLOAT TSGSL_cdf_chisq_Qinv (const RFLOAT Q, const RFLOAT nu)
{
    return gsl_cdf_chisq_Qinv(Q, nu); 
}

RFLOAT TSGSL_cdf_gaussian_Qinv (const RFLOAT Q, const RFLOAT sigma)
{
    return gsl_cdf_gaussian_Qinv(Q, sigma);
}

RFLOAT TSGSL_complex_abs2 (gsl_complex z)
{
    return gsl_complex_abs2(z); 
}

int TSGSL_fit_linear (const RFLOAT * x, const size_t xstride, const RFLOAT * y, const size_t ystride, const size_t n, RFLOAT * c0, RFLOAT * c1, RFLOAT * cov00, RFLOAT * cov01, RFLOAT * cov11, RFLOAT * sumsq)
{
    return gsl_fit_linear (x, xstride, y, ystride,  n,  c0, c1, cov00, cov01, cov11, sumsq);
}


int TSGSL_isinf (const RFLOAT x)
{
    return gsl_isinf(x); 
}

int TSGSL_isnan (const RFLOAT x)
{

    return gsl_isnan(x); 
}

RFLOAT TSGSL_pow_2(const RFLOAT x)
{
    return gsl_pow_2(x);
}

RFLOAT TSGSL_pow_3(const RFLOAT x)
{
    return  gsl_pow_3(x);
}

RFLOAT TSGSL_pow_4(const RFLOAT x)
{
    return gsl_pow_4(x);
}

void TSGSL_ran_bivariate_gaussian (const gsl_rng * r, RFLOAT sigma_x, RFLOAT sigma_y, RFLOAT rho, RFLOAT *x, RFLOAT *y)
{
    return gsl_ran_bivariate_gaussian (r,  sigma_x,  sigma_y,  rho,  x,  y);
}

void TSGSL_ran_dir_2d (const gsl_rng * r, RFLOAT * x, RFLOAT * y)
{
    return gsl_ran_dir_2d(r, x, y);;
}

RFLOAT TSGSL_ran_flat (const gsl_rng * r, const RFLOAT a, const RFLOAT b)
{
    return gsl_ran_flat(r, a, b);
}

RFLOAT TSGSL_ran_gaussian (const gsl_rng * r, const RFLOAT sigma)
{
    return gsl_ran_gaussian (r, sigma);
}

void TSGSL_ran_shuffle (const gsl_rng * r, void * base, size_t nmembm, size_t size)
{

    return gsl_ran_shuffle (r, base, nmembm, size);
}

gsl_rng *TSGSL_rng_alloc (const gsl_rng_type * T)
{
    return gsl_rng_alloc (T);
}

void TSGSL_rng_free (gsl_rng * r)
{
    return gsl_rng_free(r); 
}

unsigned long int TSGSL_rng_get (const gsl_rng * r)
{
    return gsl_rng_get (r);
}


void TSGSL_rng_set (const gsl_rng * r, unsigned long int seed)
{
    return gsl_rng_set (r, seed);
}


RFLOAT TSGSL_rng_uniform (const gsl_rng * r)
{
    return gsl_rng_uniform ( r);
}

unsigned long int TSGSL_rng_uniform_int (const gsl_rng * r, unsigned long int n)
{
    return gsl_rng_uniform_int ( r,  n);
}


RFLOAT TSGSL_sf_bessel_I0(const RFLOAT x)
{
    return gsl_sf_bessel_I0( x);
}

RFLOAT TSGSL_sf_bessel_In(const int n, const RFLOAT x)
{
    return gsl_sf_bessel_In(n,  x);
}


RFLOAT TSGSL_sf_bessel_Inu(RFLOAT nu, RFLOAT x)
{
    return gsl_sf_bessel_Inu(nu, x);
}


RFLOAT TSGSL_sf_bessel_j0(const RFLOAT x)
{
    return gsl_sf_bessel_j0( x);
}


RFLOAT TSGSL_sf_bessel_Jnu(const RFLOAT nu, const RFLOAT x)
{
    return gsl_sf_bessel_Jnu( nu,  x);
}


RFLOAT TSGSL_sf_sinc(const RFLOAT x)
{
    return gsl_sf_sinc( x);
}


void TSGSL_sort (RFLOAT * data, const size_t stride, const size_t n)
{
    return gsl_sort ( data, stride, n);
}

int TSGSL_sort_largest (RFLOAT * dest, const size_t k, const RFLOAT * src, const size_t stride, const size_t n)
{
    return gsl_sort_largest ( dest, k,  src, stride, n);
}


RFLOAT TSGSL_stats_max (const RFLOAT data[], const size_t stride, const size_t n)
{
    return gsl_stats_max ( data,  stride,  n);
}


RFLOAT TSGSL_stats_mean (const RFLOAT data[], const size_t stride, const size_t n)
{
    return gsl_stats_mean ( data,  stride,  n);
}


RFLOAT TSGSL_stats_min (const RFLOAT data[], const size_t stride, const size_t n)
{
    return gsl_stats_min ( data,  stride,  n);
}
RFLOAT TSGSL_stats_quantile_from_sorted_data (const RFLOAT sorted_data[], const size_t stride, const size_t n, const RFLOAT f)
{
    return gsl_stats_quantile_from_sorted_data ( sorted_data,  stride,  n,  f);
} 


RFLOAT TSGSL_stats_sd (const RFLOAT data[], const size_t stride, const size_t n)
{
    return gsl_stats_sd ( data,  stride,  n);
}


RFLOAT TSGSL_stats_sd_m (const RFLOAT data[], const size_t stride, const size_t n, const RFLOAT mean)
{
    return gsl_stats_sd_m ( data,  stride,  n,  mean);
}


int TSFFTW_init_threads()
{
	return fftw_init_threads();
}
void TSFFTW_cleanup_threads(void)
{
	fftw_cleanup_threads();
}
void TSFFTW_destroy_plan(TSFFTW_PLAN plan)
{
	fftw_destroy_plan(plan);
}
void TSFFTW_execute(const TSFFTW_PLAN plan)
{
	fftw_execute(plan);
}
void TSFFTW_execute_split_dft_r2c( const TSFFTW_PLAN p, RFLOAT *in, RFLOAT *ro, RFLOAT *io)
{
	fftw_execute_split_dft_r2c( p, in, ro, io);
}
void TSFFTW_execute_dft_r2c( const TSFFTW_PLAN p, RFLOAT *in, TSFFTW_COMPLEX *out)
{
    fftw_execute_dft_r2c( p, in, out);
}
void TSFFTW_execute_dft_c2r( const TSFFTW_PLAN p, TSFFTW_COMPLEX *in, RFLOAT *out)
{
	fftw_execute_dft_c2r( p, in, out);
} 
void *TSFFTW_malloc(size_t n)
{
	return fftw_malloc(n);
}
void TSFFTW_free(void *p)
{
	fftw_free(p);
}

TSFFTW_PLAN TSFFTW_plan_dft_r2c_2d(int n0, int n1, RFLOAT *in, TSFFTW_COMPLEX *out, unsigned flags)
{
	return fftw_plan_dft_r2c_2d(n0, n1, in, out, flags);
}
TSFFTW_PLAN TSFFTW_plan_dft_r2c_3d(int n0, int n1, int n2, RFLOAT *in, TSFFTW_COMPLEX *out, unsigned flags)
{
	return fftw_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
}

TSFFTW_PLAN TSFFTW_plan_dft_c2r_2d(int n0, int n1, TSFFTW_COMPLEX *in, RFLOAT *out, unsigned flags)
{
	return fftw_plan_dft_c2r_2d(n0, n1, in, out, flags);
}
TSFFTW_PLAN TSFFTW_plan_dft_c2r_3d(int n0, int n1, int n2, TSFFTW_COMPLEX *in, RFLOAT *out, unsigned flags)
{
	return fftw_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
}

void TSFFTW_plan_with_nthreads(int nthreads)
{
	fftw_plan_with_nthreads(nthreads);
}

void TSFFTW_set_timelimit(RFLOAT seconds)
{
	fftw_set_timelimit(seconds);
}

