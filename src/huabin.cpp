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
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fit.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>



void gsl_ran_float_dir_2d (const gsl_rng * r, float *x, float *y)
{
  /* This method avoids trig, but it does take an average of 8/pi =
   * 2.55 calls to the RNG, instead of one for the direct
   * trigonometric method.  */

  float u, v, s;
  do
    {
      u = -1 + 2 * gsl_rng_uniform (r);
      v = -1 + 2 * gsl_rng_uniform (r);
      s = u * u + v * v;
    }
  while (s > 1.0 || s == 0.0);

  /* This is the Von Neumann trick. See Knuth, v2, 3rd ed, p140
   * (exercise 23).  Note, no sin, cos, or sqrt !  */

  *x = (u * u - v * v) / s;
  *y = 2 * u * v / s;

  /* Here is the more straightforward approach, 
   *     s = sqrt (s);  *x = u / s;  *y = v / s;
   * It has fewer total operations, but one of them is a sqrt */
}

void gsl_ran_float_dir_2d_trig_method (const gsl_rng * r, float *x, float *y)
{
  /* This is the obvious solution... */
  /* It ain't clever, but since sin/cos are often hardware accelerated,
   * it can be faster -- it is on my home Pentium -- than von Neumann's
   * solution, or slower -- as it is on my Sun Sparc 20 at work
   */
  float t = 6.2831853071795864 * gsl_rng_uniform (r);          /* 2*PI */
  *x = cos (t);
  *y = sin (t);
}

void gsl_ran_float_dir_3d (const gsl_rng * r, float *x, float *y, float *z)
{
  float s, a;

  /* This is a variant of the algorithm for computing a random point
   * on the unit sphere; the algorithm is suggested in Knuth, v2,
   * 3rd ed, p136; and attributed to Robert E Knop, CACM, 13 (1970),
   * 326.
   */

  /* Begin with the polar method for getting x,y inside a unit circle
   */
  do
    {
      *x = -1 + 2 * gsl_rng_uniform (r);
      *y = -1 + 2 * gsl_rng_uniform (r);
      s = (*x) * (*x) + (*y) * (*y);
    }
  while (s > 1.0);

  *z = -1 + 2 * s;              /* z uniformly distributed from -1 to 1 */
  a = 2 * sqrt (1 - s);         /* factor to adjust x,y so that x^2+y^2
                                 * is equal to 1-z^2 */
  *x *= a;
  *y *= a;
}

void gsl_ran_float_dir_nd (const gsl_rng * r, size_t n, float *x)
{
  float d;
  size_t i;
  /* See Knuth, v2, 3rd ed, p135-136.  The method is attributed to
   * G. W. Brown, in Modern Mathematics for the Engineer (1956).
   * The idea is that gaussians G(x) have the property that
   * G(x)G(y)G(z)G(...) is radially symmetric, a function only
   * r = sqrt(x^2+y^2+...)
   */
  d = 0;
  do
    {
      for (i = 0; i < n; ++i)
        {
          x[i] = gsl_ran_gaussian (r, 1.0);
          d += x[i] * x[i];
        }
    }
  while (d == 0);
  d = sqrt (d);
  for (i = 0; i < n; ++i)
    {
      x[i] /= d;
    }
}
void gsl_ran_float_bivariate_gaussian (const gsl_rng * r, float sigma_x, float sigma_y, float rho, float *x, float *y)
{
  float u, v, r2, scale;

  do
    {
      /* choose x,y in uniform square (-1,-1) to (+1,+1) */

      u = -1 + 2 * gsl_rng_uniform (r);
      v = -1 + 2 * gsl_rng_uniform (r);

      /* see if it is in the unit circle */
      r2 = u * u + v * v;
    }
  while (r2 > 1.0 || r2 == 0);

  scale = sqrt (-2.0 * log (r2) / r2);

  *x = sigma_x * u * scale;
  *y = sigma_y * (rho * u + sqrt(1 - rho*rho) * v) * scale;
}

float gsl_ran_float_bivariate_gaussian_pdf (const float x, const float y, const float sigma_x, const float sigma_y, const float rho)
{
  float u = x / sigma_x ;
  float v = y / sigma_y ;
  float c = 1 - rho*rho ;
  float p = (1 / (2 * M_PI * sigma_x * sigma_y * sqrt(c))) 
    * exp (-(u * u - 2 * rho * u * v + v * v) / (2 * c));
  return p;
}

int gsl_fit_float_linear (const float *x, const size_t xstride, const float *y, const size_t ystride, const size_t n, float *c0, float *c1, float *cov_00, float *cov_01, float *cov_11, float *sumsq)
{
  float m_x = 0, m_y = 0, m_dx2 = 0, m_dxdy = 0;

  size_t i;

  for (i = 0; i < n; i++)
    {
      m_x += (x[i * xstride] - m_x) / (i + 1.0);
      m_y += (y[i * ystride] - m_y) / (i + 1.0);
    }

  for (i = 0; i < n; i++)
    {
      const float dx = x[i * xstride] - m_x;
      const float dy = y[i * ystride] - m_y;

      m_dx2 += (dx * dx - m_dx2) / (i + 1.0);
      m_dxdy += (dx * dy - m_dxdy) / (i + 1.0);
    }

  /* In terms of y = a + b x */

  {
    float s2 = 0, d2 = 0;
    float b = m_dxdy / m_dx2;
    float a = m_y - m_x * b;

    *c0 = a;
    *c1 = b;

    /* Compute chi^2 = \sum (y_i - (a + b * x_i))^2 */

    for (i = 0; i < n; i++)
      {
        const float dx = x[i * xstride] - m_x;
        const float dy = y[i * ystride] - m_y;
        const float d = dy - b * dx;
        d2 += d * d;
      }

    s2 = d2 / (n - 2.0);        /* chisq per degree of freedom */

    *cov_00 = s2 * (1.0 / n) * (1 + m_x * m_x / m_dx2);
    *cov_11 = s2 * 1.0 / (n * m_dx2);

    *cov_01 = s2 * (-m_x) / (n * m_dx2);

    *sumsq = d2;
  }

  return GSL_SUCCESS;
}





RFLOAT TSGSL_cdf_chisq_Qinv (const RFLOAT Q, const RFLOAT nu)
{
    return gsl_cdf_chisq_Qinv(Q, nu); 
}

RFLOAT TSGSL_cdf_gaussian_Qinv (const RFLOAT Q, const RFLOAT sigma)
{
    return gsl_cdf_gaussian_Qinv(Q, sigma);
}

RFLOAT TSGSL_complex_abs2 (Complex z)
{
    RFLOAT x = z.dat[0];
    RFLOAT y = z.dat[1];

    return (x * x + y * y);

    /*
     *return gsl_complex_abs2(z); 
     */
}

int TSGSL_fit_linear (const RFLOAT * x, const size_t xstride, const RFLOAT * y, const size_t ystride, const size_t n, RFLOAT * c0, RFLOAT * c1, RFLOAT * cov00, RFLOAT * cov01, RFLOAT * cov11, RFLOAT * sumsq)
{
#ifdef USING_SINGLE_PRECISION
    return gsl_fit_float_linear (x, xstride, y, ystride,  n,  c0, c1, cov00, cov01, cov11, sumsq);
#else
    return gsl_fit_linear (x, xstride, y, ystride,  n,  c0, c1, cov00, cov01, cov11, sumsq);
#endif
}



/*
 *RFLOAT TSGSL_MAX_RFLOAT(RFLOAT a, RFLOAT b)
 *{
 *    return  a > b ? a : b;
 *}
 *
 *RFLOAT TSGSL_MIN_RFLOAT(RFLOAT a, RFLOAT b)
 *{
 *    return a < b ? a: b;
 *}
 *
 */
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
#ifdef USING_SINGLE_PRECISION
    return gsl_ran_float_bivariate_gaussian (r,  sigma_x,  sigma_y,  rho,  x,  y);
#else
    return gsl_ran_bivariate_gaussian (r,  sigma_x,  sigma_y,  rho,  x,  y);
#endif
}

void TSGSL_ran_dir_2d (const gsl_rng * r, RFLOAT * x, RFLOAT * y)
{
#ifdef USING_SINGLE_PRECISION
    return gsl_ran_float_dir_2d(r, x, y);;
#else
    return gsl_ran_dir_2d(r, x, y);;
#endif
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
#ifdef USING_SINGLE_PRECISION
    gsl_sort_float(data, stride, n);
#else
    gsl_sort ( data, stride, n);
#endif
}

int TSGSL_sort_largest (RFLOAT * dest, const size_t k, const RFLOAT * src, const size_t stride, const size_t n)
{
#ifdef USING_SINGLE_PRECISION
    return gsl_sort_float_largest ( dest, k,  src, stride, n);
#else
    return gsl_sort_largest ( dest, k,  src, stride, n);
#endif
}


RFLOAT TSGSL_stats_max (const RFLOAT data[], const size_t stride, const size_t n)
{
#ifdef USING_SINGLE_PRECISION
    return gsl_stats_float_max(data, stride, n);
#else
    return gsl_stats_max ( data,  stride,  n);
#endif
}


RFLOAT TSGSL_stats_mean (const RFLOAT data[], const size_t stride, const size_t n)
{
#ifdef USING_SINGLE_PRECISION
    return gsl_stats_float_mean(data, stride, n);
#else
    return gsl_stats_mean ( data,  stride,  n);
#endif
}


RFLOAT TSGSL_stats_min (const RFLOAT data[], const size_t stride, const size_t n)
{
#ifdef USING_SINGLE_PRECISION
    return gsl_stats_float_min(data, stride, n);
#else
    return gsl_stats_min ( data,  stride,  n);
#endif
}

RFLOAT TSGSL_stats_quantile_from_sorted_data (const RFLOAT sorted_data[], const size_t stride, const size_t n, const RFLOAT f)
{
#ifdef USING_SINGLE_PRECISION
    return gsl_stats_float_quantile_from_sorted_data(sorted_data,  stride,  n,  f);
#else
    return gsl_stats_quantile_from_sorted_data ( sorted_data,  stride,  n,  f);
#endif
} 


RFLOAT TSGSL_stats_sd (const RFLOAT data[], const size_t stride, const size_t n)
{
#ifdef USING_SINGLE_PRECISION
    return gsl_stats_float_sd(data, stride, n);
#else
    return gsl_stats_sd ( data,  stride,  n);
#endif
}


RFLOAT TSGSL_stats_sd_m (const RFLOAT data[], const size_t stride, const size_t n, const RFLOAT mean)
{
#ifdef USING_SINGLE_PRECISION
    return gsl_stats_float_sd_m(data, stride, n, mean);
#else
    return gsl_stats_sd_m ( data,  stride,  n,  mean);
#endif
}


int TSFFTW_init_threads()
{
#ifdef USING_SINGLE_PRECISION
	return fftwf_init_threads();
#else
	return fftw_init_threads();
#endif
}
void TSFFTW_cleanup_threads(void)
{
#ifdef USING_SINGLE_PRECISION
	fftwf_cleanup_threads();
#else
	fftw_cleanup_threads();
#endif
}
void TSFFTW_destroy_plan(TSFFTW_PLAN plan)
{
#ifdef USING_SINGLE_PRECISION
	fftwf_destroy_plan(plan);
#else
    fftw_destroy_plan(plan);
#endif
}
void TSFFTW_execute(const TSFFTW_PLAN plan)
{
#ifdef USING_SINGLE_PRECISION
	fftwf_execute(plan);
#else
	fftw_execute(plan);
#endif
}
void TSFFTW_execute_split_dft_r2c( const TSFFTW_PLAN p, RFLOAT *in, RFLOAT *ro, RFLOAT *io)
{
    
#ifdef USING_SINGLE_PRECISION
	fftwf_execute_split_dft_r2c( p, in, ro, io);
#else
	fftw_execute_split_dft_r2c( p, in, ro, io);
#endif
}
void TSFFTW_execute_dft_r2c( const TSFFTW_PLAN p, RFLOAT *in, TSFFTW_COMPLEX *out)
{
#ifdef USING_SINGLE_PRECISION
    fftwf_execute_dft_r2c( p, in, out);
#else
    fftw_execute_dft_r2c( p, in, out);
#endif
}
void TSFFTW_execute_dft_c2r( const TSFFTW_PLAN p, TSFFTW_COMPLEX *in, RFLOAT *out)
{
#ifdef USING_SINGLE_PRECISION
	fftwf_execute_dft_c2r( p, in, out);
#else
	fftw_execute_dft_c2r( p, in, out);
#endif
} 
void *TSFFTW_malloc(size_t n)
{
#ifdef USING_SINGLE_PRECISION
    return fftwf_malloc(n);
#else
    return fftw_malloc(n);
#endif
}
void TSFFTW_free(void *p)
{
#ifdef USING_SINGLE_PRECISION
    fftwf_free(p);
#else
    fftw_free(p);
#endif
}

TSFFTW_PLAN TSFFTW_plan_dft_r2c_2d(int n0, int n1, RFLOAT *in, TSFFTW_COMPLEX *out, unsigned flags)
{
#ifdef USING_SINGLE_PRECISION
	return fftwf_plan_dft_r2c_2d(n0, n1, in, out, flags);
#else
	return fftw_plan_dft_r2c_2d(n0, n1, in, out, flags);
#endif
}
TSFFTW_PLAN TSFFTW_plan_dft_r2c_3d(int n0, int n1, int n2, RFLOAT *in, TSFFTW_COMPLEX *out, unsigned flags)
{
#ifdef USING_SINGLE_PRECISION
	return fftwf_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
#else
	return fftw_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
#endif
}

TSFFTW_PLAN TSFFTW_plan_dft_c2r_2d(int n0, int n1, TSFFTW_COMPLEX *in, RFLOAT *out, unsigned flags)
{
#ifdef USING_SINGLE_PRECISION
	return fftwf_plan_dft_c2r_2d(n0, n1, in, out, flags);
#else
	return fftw_plan_dft_c2r_2d(n0, n1, in, out, flags);
#endif
}
TSFFTW_PLAN TSFFTW_plan_dft_c2r_3d(int n0, int n1, int n2, TSFFTW_COMPLEX *in, RFLOAT *out, unsigned flags)
{
#ifdef USING_SINGLE_PRECISION
	return fftwf_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
#else
	return fftw_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
#endif
}

void TSFFTW_plan_with_nthreads(int nthreads)
{
#ifdef USING_SINGLE_PRECISION
	fftwf_plan_with_nthreads(nthreads);
#else
	fftw_plan_with_nthreads(nthreads);
#endif
}

void TSFFTW_set_timelimit(RFLOAT seconds)
{
#ifdef USING_SINGLE_PRECISION
	fftwf_set_timelimit(seconds);
#else
	fftw_set_timelimit(seconds);
#endif
}

