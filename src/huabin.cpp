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

double TSGSL_cdf_chisq_Qinv (const double Q, const double nu)
{
    return gsl_cdf_chisq_Qinv(Q, nu); 
}

double TSGSL_cdf_gaussian_Qinv (const double Q, const double sigma)
{
    return gsl_cdf_gaussian_Qinv(Q, sigma);
}

double TSGSL_complex_abs2 (gsl_complex z)
{
    return gsl_complex_abs2(z); 
}

int TSGSL_fit_linear (const double * x, const size_t xstride, const double * y, const size_t ystride, const size_t n, double * c0, double * c1, double * cov00, double * cov01, double * cov11, double * sumsq)
{
    return gsl_fit_linear (x, xstride, y, ystride,  n,  c0, c1, cov00, cov01, cov11, sumsq);
}


int TSGSL_isinf (const double x)
{
    return gsl_isinf(x); 
}

int TSGSL_isnan (const double x)
{

    return gsl_isnan(x); 
}

double TSGSL_pow_2(const double x)
{
    return gsl_pow_2(x);
}

double TSGSL_pow_3(const double x)
{
    return  gsl_pow_3(x);
}

double TSGSL_pow_4(const double x)
{
    return gsl_pow_4(x);
}

void TSGSL_ran_bivariate_gaussian (const gsl_rng * r, double sigma_x, double sigma_y, double rho, double *x, double *y)
{
    return gsl_ran_bivariate_gaussian (r,  sigma_x,  sigma_y,  rho,  x,  y);
}

void TSGSL_ran_dir_2d (const gsl_rng * r, double * x, double * y)
{
    return gsl_ran_dir_2d(r, x, y);;
}

double TSGSL_ran_flat (const gsl_rng * r, const double a, const double b)
{
    return gsl_ran_flat(r, a, b);
}

double TSGSL_ran_gaussian (const gsl_rng * r, const double sigma)
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


double TSGSL_rng_uniform (const gsl_rng * r)
{
    return gsl_rng_uniform ( r);
}

unsigned long int TSGSL_rng_uniform_int (const gsl_rng * r, unsigned long int n)
{
    return gsl_rng_uniform_int ( r,  n);
}


double TSGSL_sf_bessel_I0(const double x)
{
    return gsl_sf_bessel_I0( x);
}

double TSGSL_sf_bessel_In(const int n, const double x)
{
    return gsl_sf_bessel_In(n,  x);
}


double TSGSL_sf_bessel_Inu(double nu, double x)
{
    return gsl_sf_bessel_Inu(nu, x);
}


double TSGSL_sf_bessel_j0(const double x)
{
    return gsl_sf_bessel_j0( x);
}


double TSGSL_sf_bessel_Jnu(const double nu, const double x)
{
    return gsl_sf_bessel_Jnu( nu,  x);
}


double TSGSL_sf_sinc(const double x)
{
    return gsl_sf_sinc( x);
}


void TSGSL_sort (double * data, const size_t stride, const size_t n)
{
    return gsl_sort ( data, stride, n);
}

int TSGSL_sort_largest (double * dest, const size_t k, const double * src, const size_t stride, const size_t n)
{
    return gsl_sort_largest ( dest, k,  src, stride, n);
}


double TSGSL_stats_max (const double data[], const size_t stride, const size_t n)
{
    return gsl_stats_max ( data,  stride,  n);
}


double TSGSL_stats_mean (const double data[], const size_t stride, const size_t n)
{
    return gsl_stats_mean ( data,  stride,  n);
}


double TSGSL_stats_min (const double data[], const size_t stride, const size_t n)
{
    return gsl_stats_min ( data,  stride,  n);
}
double TSGSL_stats_quantile_from_sorted_data (const double sorted_data[], const size_t stride, const size_t n, const double f)
{
    return gsl_stats_quantile_from_sorted_data ( sorted_data,  stride,  n,  f);
} 


double TSGSL_stats_sd (const double data[], const size_t stride, const size_t n)
{
    return gsl_stats_sd ( data,  stride,  n);
}


double TSGSL_stats_sd_m (const double data[], const size_t stride, const size_t n, const double mean)
{
    return gsl_stats_sd_m ( data,  stride,  n,  mean);
}



