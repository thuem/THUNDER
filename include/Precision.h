/** @file
 *  @author Huabin Ruan
 *  @version 1.4.11.180913
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Huabin Ruan | 2018/09/13 | 1.4.11.080913 | Add header for file and functions
 *
 *  @brief Precision.h encapsulates the header files/MACRO/data structures/functions of the single-precision version and the double-precision version
  */

#ifndef  PRECISION_H
#define  PRECISION_H

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

/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_cdf_chisq_Qinv (const RFLOAT Q, const RFLOAT nu);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_cdf_gaussian_Qinv (const RFLOAT Q, const RFLOAT sigma);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
void TSGSL_ran_bivariate_gaussian (const gsl_rng * r, RFLOAT sigma_x, RFLOAT sigma_y, RFLOAT rho, RFLOAT *x, RFLOAT *y);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
int TSGSL_fit_linear (const RFLOAT * x, const size_t xstride, const RFLOAT * y, const size_t ystride, const size_t n, RFLOAT * c0, RFLOAT * c1, RFLOAT * cov00, RFLOAT * cov01, RFLOAT * cov11, RFLOAT * sumsq);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
void TSGSL_ran_dir_2d (const gsl_rng * r, RFLOAT * x, RFLOAT * y);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_ran_flat (const gsl_rng * r, const RFLOAT a, const RFLOAT b);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_ran_gaussian (const gsl_rng * r, const RFLOAT sigma);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_sf_bessel_I0(const RFLOAT x);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_sf_bessel_In(const int n, const RFLOAT x);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_sf_bessel_Inu(RFLOAT nu, RFLOAT x);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_sf_bessel_j0(const RFLOAT x);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_sf_bessel_Jnu(const RFLOAT nu, const RFLOAT x);
/**
 *  @brief  Doctor Hu, please add the description for this function 
 */
RFLOAT TSGSL_sf_sinc(const RFLOAT x);



/**
 *  @brief Returns the trigonometric sin of an angle x with type RFLOAT, which is either of single precision or double precision, determined at compiled time.
 *  
 *  @return the sin of the argument
 */
inline RFLOAT TS_SIN(const RFLOAT x /**< [in] the argument whose sin value is to be determined. */)
{
#ifdef SINGLE_PRECISION
    return sinf(x);
#else
    return sin(x);
#endif
}


/**
 *  @brief Returns the trigonometric cosine of an angle x with type RFLOAT, which is either of single precision or double precision, determined at compiled time.
 *  
 *  @return the cosine of the argument
 */
inline RFLOAT TS_COS(const RFLOAT x /**< [in] the argument whose cosine value is to be determined */)
{
#ifdef SINGLE_PRECISION
    return cosf(x);
#else
    return cos(x);
#endif
}


/**
 *  @brief Returns the correctly rounded positive square root of x, with type of RFLOAT, which is either single precision or double precision determined at compilied time.
 *  
 *  @return the positive square root of x
 */
inline RFLOAT TS_SQRT(const RFLOAT x /**< [in] the argument whose positive square root value is to be detemined. */)

{
#ifdef SINGLE_PRECISION
    return sqrtf(x);
#else
    return sqrt(x);
#endif
}


/**
 *  @brief Returns the greater of two RFLOAT values. The type of RFLOAT is either float or double, determined at compilied time. 
 *
 *  @return the larger of a and b.
 */
inline RFLOAT TSGSL_MAX_RFLOAT(const RFLOAT a, /**< [in] an argument. */
                               const RFLOAT b  /**< [in] another argument. */
                              )
{
    return a > b ? a : b;
}

/**
 *  @brief Returns the smaller of two RFLOAT values. The type of RFLOAT is either float or double, determined at compilied time. 
 *
 *  @return the smaller of a and b.
 */
inline RFLOAT TSGSL_MIN_RFLOAT(const RFLOAT a, /**< [in] an argument. */
                               const RFLOAT b  /**< [in] another argument. */
                              )
{
    return a < b ? a : b;
}

/**
 *  @brief Returns the square of complex modulus length 
 *
 *  @return the square of modulus length of z
 */
RFLOAT TSGSL_complex_abs2 (Complex z /**< [in] the argument whose square of modulus length is determined*/);


/**
 *  @brief Determines whether x is infinity
 *
 *  @return 1 if x is positive infinity, -1 if x is negative infinity and 0 otherwise

 */
int TSGSL_isinf (const RFLOAT x /**< [in] the argment used to determined whether it is infinity. */);

/**
 *  @brief Determines whether x is a number
 *
 *  @return 1 if x is not-a-number and 0 otherwise
 */
int TSGSL_isnan (const RFLOAT x /**< [in] the argment used to determined whether it is a number. */);
;

/**
 *  @brief Calculates the value of @f$x^2@f$
 *
 *  @return The result @f$x^2@f$
 */
RFLOAT TSGSL_pow_2(const RFLOAT x /**< [in] the argument whose @f$x^2@f$ is calculated. */);

/**
 *  @brief Calculates the value of @f$x^3@f$
 *
 *  @return The result of @f$x^3@f$
 */
RFLOAT TSGSL_pow_3(const RFLOAT x /**< [in] the argument whose @f$x^3@f$ is calculated. */);

/**
 *  @brief Calculates the value of @f$x^4@f$
 *
 *  @return The result of @f$x^4@f$
 */
RFLOAT TSGSL_pow_4(const RFLOAT x /**< [in] the argument whose @f$x^4@f$ is calculated. */);



/**
 *  @brief Shuffles the data items in buffer base randomly.
 */
void TSGSL_ran_shuffle (const gsl_rng *r, /**< [in] randon engine used for randomness. */
                        void *base,       /**< [in,out] data elements to be shuffled. */
                        size_t nmembm,    /**< [in] number of data items to be shuffled. */
                        size_t size       /**< [in] size of each data items. */
                       );


/**
 *  @brief Creates an instance of a random nunber generator
 *  
 *  @return a pointer to a newly-created instance of a random number generator of type T
 */
gsl_rng *TSGSL_rng_alloc (const gsl_rng_type *T /**< [in] type of random number generator*/);

/**
 *  @brief Frees all the memory associated with the generator r
 *
 */
void TSGSL_rng_free (gsl_rng *r /**< [in] generator to be freed*/);

/**
 *  @brief Gets a random integer from the generator r. The minimum and maximum values depend on the algorithm used, but all integers in the range [min, max] are equally likely. The values of min and max can be determined using the auxiliary functions gsl_rng_max() and gsl_rng_min()
 *
 *  @return a random integer from the generator r.
 */
size_t TSGSL_rng_get (const gsl_rng *r /**< [in] a random generator. */);

/**
 *  @brief Initializes (or “seeds”) the random number generator
 */
void TSGSL_rng_set (const gsl_rng *r /**< [in] random number generator*/,
                    size_t seed      /**< [in] seed used for random number generator. */
                   );


/**
 *  @brief Gets a float point number with type of RFLOAT uniformly distributed in the range [0,1)
 *
 *  @return a floating point number with type of RFLOAT uniformly distributed in the range [0,1)
 */
RFLOAT TSGSL_rng_uniform (const gsl_rng * r /**< [in] a random engine used to generate the number. */);

/**
 *  @brief Gets a random integer from 0 to @f$n-1@f$ inclusive by scaling down and/or discarding samples from the generator @f$r@f$.
 *
 *  @return A random integer in the range [0, n)
 */
size_t TSGSL_rng_uniform_int (const gsl_rng * r, /**< [in] a random engine used to generate the number. */
                              size_t n           /**< [in] the upper bound*/
                             );


/**
 * @brief Sorts the @f$n@f$ elements of the array data with stride @f$stride@f$ into ascending numerical order 
 */
void TSGSL_sort(RFLOAT * data,/**< [in,out] data elements to be sorted. */
                 const size_t stride /**< [in] stride argument */,
                 const size_t n      /**< [in] the number of elements to be sorted*/
                );

/**
 * @brief Copies the @f$k@f$ largest elements of the array @f$src@f$, of size @f$n@f$ and stride @f$stride@f$, in descending numerical order into the array @f$dest@f$. @f$k@f$ must be less than or equal to @f$n@f$. The data @f$src@f$ is not modified by this operation.
 *
 * @return 1 if copy success, or 0 if copy failed 
 */
int TSGSL_sort_largest(RFLOAT * dst,        /**< [out] destination buffer. */
                       const size_t k,      /**< [in]  number of elements to be copied. */
                       const RFLOAT * src,  /**< [in]  source buffer. */
                       const size_t stride, /**< [in]  stride argument. */
                       const size_t n       /**< [in]  total number of source elements. */
                      );


/**
 *  @brief Sorts the @f$n@f$ elements of the array @f$src@f$ with stride @f$stride@f$ into ascending order, storing the resulting permutation in @f$dst@f$. The elements of @f$dst@f$ give the index of the vector element. 
 */
void TSGSL_sort_index(size_t * dst,        /**< [out] destination buffer. */
                      const RFLOAT * src,  /**< [in]  source buffer. */
                      const size_t stride, /**< [in]  stride argument. */
                      const size_t n       /**< [in]  total number of source elements. */
                     );

        
/**
 *  @brief Stores the indices of the @f$k@f$ smallest elements of the array @f$src@f$, of size @f$n@f$ and stride @f$stride@f$, in the array @f$p@f$.
 */
void TSGSL_sort_smallest_index(size_t* dst,         /**< [out] destination buffer. */               
                               const size_t k,      /**< [in]  number of elements to be choosed. */
                               const RFLOAT* src,   /**< [in]  source buffer. */
                               const size_t stride, /**< [in]  stride argument. */
                               const size_t n       /**< [in]  total number of source elements. */
                              );

/**
 *  @brief Stores the indices of the @f$k@f$ largest elements of the array @f$src@f$, of size @f$n@f$ and stride @f$stride@f$, in the array @f$p@f$.
 */
void TSGSL_sort_largest_index(size_t* dst,          /**< [out] destination buffer. */               
                               const size_t k,      /**< [in]  number of elements to be choosed. */
                               const RFLOAT* src,   /**< [in]  source buffer. */
                               const size_t stride, /**< [in]  stride argument. */
                               const size_t n       /**< [in]  total number of source elements. */
                              );


/**
 *  @brief Gets the maximum value in @f$data@f$, a dataset of length @f$n@f$ with stride @f$stride@f$.
 *
 *  @return the maximum value with stride @f$stride@f$.
 */
RFLOAT TSGSL_stats_max(const RFLOAT data[], /**< [in] source data buffer. */
                       const size_t stride, /**< [in] stride argument. */
                       const size_t n       /**< [in] total number of source elements. */
                      );
/**
 *  @brief Gets the minimum value in @f$data@f$, a dataset of length @f$n@f$ with stride @f$stride@f$.
 *
 *  @return the minimum value with stride @f$stride@f$.
 */
RFLOAT TSGSL_stats_min(const RFLOAT data[],  /**< [in] source data buffer. */
                        const size_t stride, /**< [in] stride argument. */
                        const size_t n       /**< [in] total number of source elements. */
                      );
/**
 *  @brief Gets the mean value in @f$data@f$, a dataset of length @f$n@f$ with stride @f$stride@f$.
 *
 *  @return the mean value with stride @f$stride@f$.
 */
RFLOAT TSGSL_stats_mean(const RFLOAT data[], /**< [in] source data buffer. */
                       const size_t stride,  /**< [in] stride argument. */
                       const size_t n        /**< [in] total number of source elements. */
                      );


/**
 *  @brief
 */
RFLOAT TSGSL_stats_quantile_from_sorted_data(const RFLOAT sorted_data[], 
                                             const size_t stride, 
                                             const size_t n, 
                                             const RFLOAT f
                                            );

/**
 *  @brief
 */
RFLOAT TSGSL_stats_sd(const RFLOAT data[],
                      const size_t stride,
                      const size_t n
                     );

/**
 *  @brief
 */
RFLOAT TSGSL_stats_sd_m(const RFLOAT data[],
                        const size_t stride,
                        const size_t n,
                        const RFLOAT mean
                       );

/**
 *  @brief
 */
int TSFFTW_init_threads();

/**
 *  @brief
 */
void TSFFTW_cleanup_threads();

/**
 *  @brief
 */
void TSFFTW_destroy_plan(TSFFTW_PLAN plan);

/**
 *  @brief
 */
void TSFFTW_execute(const TSFFTW_PLAN plan);

/**
 *  @brief
 */
void TSFFTW_execute_dft_r2c(const TSFFTW_PLAN p,
                            RFLOAT *in, 
                            TSFFTW_COMPLEX *out
                           );

/**
 *  @brief
 */
void TSFFTW_execute_dft_c2r(const TSFFTW_PLAN p,
                            TSFFTW_COMPLEX *in,
                            RFLOAT *out
                           ); 

/**
 *  @brief
 */
void *TSFFTW_malloc(size_t n);

/**
 *  @brief
 */
void TSFFTW_free(void *p);

/**
 *  @brief
 */
TSFFTW_PLAN TSFFTW_plan_dft_r2c_2d(int n0,
                                   int n1,
                                   RFLOAT *in,
                                   TSFFTW_COMPLEX *out,
                                   unsigned flags
                                  );

/**
 *  @brief
 */
TSFFTW_PLAN TSFFTW_plan_dft_r2c_3d(int n0,
                                   int n1,
                                   int n2,
                                   RFLOAT *in,
                                   TSFFTW_COMPLEX *out,
                                   unsigned flags
                                  );

/**
 *  @brief
 */
TSFFTW_PLAN TSFFTW_plan_dft_c2r_2d(int n0,
                                   int n1,
                                   TSFFTW_COMPLEX *in,
                                   RFLOAT *out,
                                   unsigned flags
                                  );

/**
 *  @brief
 */
TSFFTW_PLAN TSFFTW_plan_dft_c2r_3d(int n0,
                                   int n1,
                                   int n2,
                                   TSFFTW_COMPLEX *in,
                                   RFLOAT *out,
                                   unsigned flags
                                  );

/**
 *  @brief
 */
void TSFFTW_plan_with_nthreads(int nthreads);

/**
 *  @brief
 */
void TSFFTW_set_timelimit(RFLOAT seconds);

#endif // PRECISION_H
