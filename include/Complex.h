//This header file is add by huabin
#include "huabin.h"
/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef COMPLEX_H
#define COMPLEX_H

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <math.h>

#include "Config.h"

#include "Typedef.h"

/*
 *#define CONJUGATE(a) gsl_complex_conjugate(a)
 *
 *#define ABS(a) gsl_complex_abs(a)
 *
 *#define ABS2(a) gsl_complex_abs2(a)
 *
 *#define COMPLEX_POLAR(phi) gsl_complex_polar(1, phi)
 *
 *#define COMPLEX(a, b) gsl_complex_rect(a, b)
 *
 *#define REAL(a) GSL_REAL(a)
 *
 *#define IMAG(a) GSL_IMAG(a)
 *
 */
#define COMPLEX_POLAR(phi) ts_complex_polar(1.0, phi)

inline Complex ts_complex_polar(RFLOAT r, RFLOAT phi)
{
#ifdef SINGLE_PRECISION
    Complex z;
    z.dat[0] = r * cosf(phi);
    z.dat[1] = r * sinf(phi);
#else
    Complex z;
    z.dat[0] = r * cos(phi);
    z.dat[1] = r * sin(phi);
#endif
    return z;
};

inline Complex CONJUGATE(const Complex &a)
{
    Complex z;
    z.dat[0] = a.dat[0];
    z.dat[1] = -a.dat[1];
    return z;
}

static RFLOAT ts_hypot (const RFLOAT x, const RFLOAT y)
{
#ifdef SINGLE_PRECISION
  RFLOAT xabs = fabsf(x) ;
  RFLOAT yabs = fabsf(y) ;
#else
  RFLOAT xabs = fabs(x) ;
  RFLOAT yabs = fabs(y) ;
#endif


  RFLOAT min, max;

  if (xabs < yabs) {
    min = xabs ;
    max = yabs ;
  } else {
    min = yabs ;
    max = xabs ;
  }

  if (min == 0) 
    {
      return max ;
    }

  {
    RFLOAT u = min / max ;
#ifdef SINGLE_PRECISION
    return max * sqrtf (1 + u * u) ;
#else
    return max * sqrt (1 + u * u) ;
#endif
  }
};


inline RFLOAT ABS(const Complex &a)
{
    return ts_hypot(a.dat[0], a.dat[1]);

};

inline RFLOAT ABS2(const Complex &a)
{
    RFLOAT result = a.dat[0] * a.dat[0] + a.dat[1] * a.dat[1];
    return result;
};

inline Complex COMPLEX(RFLOAT a, RFLOAT b)
{
    Complex z;
    z.dat[0] = a;
    z.dat[1] = b;
    return z;
};

inline RFLOAT REAL(const Complex& a)
{
    return a.dat[0];
};
inline RFLOAT IMAG(const Complex& a)
{
    return a.dat[1];
};

inline RFLOAT gsl_real(const Complex& a)
{
    return a.dat[0];
};

inline RFLOAT gsl_imag(const Complex& a)
{
    return a.dat[1];
};

inline RFLOAT gsl_real_imag_sum(const Complex& a)
{
    return a.dat[0] + a.dat[1];
};

inline Complex operator-(const Complex& a)
{
    //return COMPLEX(-REAL(a), -IMAG(a));
    Complex newa;
    newa.dat[0] =  -a.dat[0];
    newa.dat[1] =  -a.dat[1];
    return newa;
};

inline Complex operator+(const Complex& a, const Complex& b)
{
    //return gsl_complex_add(a, b);
    Complex result;
    result.dat[0] = a.dat[0] + b.dat[0];
    result.dat[1] = a.dat[1] + b.dat[1];
    return result;
};

inline Complex operator-(const Complex& a, const Complex& b)
{
    Complex result;
    result.dat[0] = a.dat[0] - b.dat[0];
    result.dat[1] = a.dat[1] - b.dat[1];
    return result;

};

inline Complex operator*(const Complex& a, const Complex& b)
{
    Complex result;
    /*
     *result.dat[0] = a.dat[0] * b.dat[0];
     *result.dat[1] = a.dat[1] * b.dat[1];
     */
    result.dat[0] = a.dat[0] * b.dat[0] - a.dat[1] * b.dat[1];
    result.dat[1] = a.dat[0] * b.dat[1] + a.dat[1] * b.dat[0];
    return result;

};

inline Complex operator/(const Complex& a, const Complex& b)
{
    //return gsl_complex_div(a, b);
    /* 
     RFLOAT cd = op.norm();
     RFLOAT realval = real*op.real + imag*op.imag;
     RFLOAT imagval = imag*op.real - real*op.imag;
     return Complex(realval/cd, imagval/cd);*


    RFLOAT Complex::norm()
    {
        return real*real + imag*imag;
    }
    RFLOAT norm(const Complex& op)
    {
        return op.real*op.real + op.imag*op.imag;
    }
     * */
    Complex result;
    result.dat[0] = a.dat[0] * b.dat[0] + a.dat[1] * b.dat[1];
    result.dat[1] = a.dat[1] * b.dat[0] - a.dat[0] * b.dat[1]; 
    

    RFLOAT norm = b.dat[0] * b.dat[0] + b.dat[1] * b.dat[1];
    result.dat[0] /= norm;
    result.dat[1] /= norm;
    /*
     *result.dat[0] = a.dat[0] / b.dat[0];
     *result.dat[1] = a.dat[1] / b.dat[1];
     */
    return result;

};

inline void operator+=(Complex& a, const Complex b) { a = a + b; };

inline void operator-=(Complex& a, const Complex b) { a = a - b; };

inline void operator*=(Complex& a, const Complex b) { a = a * b; };

inline void operator/=(Complex& a, const Complex b) { a = a / b; };

inline Complex operator*(const Complex a, const RFLOAT x)
{
    /*
     *return gsl_complex_mul_real(a, x);
     */
    Complex result;
    result.dat[0] = a.dat[0] * x;
    result.dat[1] = a.dat[1] * x;
    return result;

};

inline Complex operator*(const RFLOAT x, const Complex a)
{
    /*
     *return a * x;
     */
    Complex result;
    result.dat[0] = a.dat[0] * x;
    result.dat[1] = a.dat[1] * x;
    return result;

};

inline void operator*=(Complex& a, const RFLOAT x)
{
    /*
     *a = a * x;
     */
    a.dat[0] = a.dat[0] * x;
    a.dat[1] = a.dat[1] * x;
};

inline Complex operator/(const Complex a, const RFLOAT x)
{
    /*
     *return gsl_complex_div_real(a, x);
     */
    Complex result;
    result.dat[0] = a.dat[0] / x;
    result.dat[1] = a.dat[1] / x;
    return result;

};

inline void operator/=(Complex& a, const RFLOAT x)
{
    /*
     *a = a / x; 
     */
    a.dat[0] = a.dat[0] / x;
    a.dat[1] = a.dat[1] / x;

};

#endif // COMPLEX_H
