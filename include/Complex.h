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
inline Complex ts_complex_polar(float r, float phi)
{
    Complex z;
    z.dat[0] = r * cosf(phi);
    z.dat[1] = r * sinf(phi);

    return z;
};

inline Complex CONJUGATE(const Complex &a)
{
    Complex z;
    z.dat[0] = a.dat[0];
    z.dat[1] = 0.0f - a.dat[1];
    return z;
}

static float ts_hypot (const float x, const float y)
{
  float xabs = fabsf(x) ;
  float yabs = fabsf(y) ;
  float min, max;

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
    float u = min / max ;
    return max * sqrtf (1 + u * u) ;
  }
};


inline float ABS(const Complex &a)
{
    return ts_hypot(a.dat[0], a.dat[1]);

};

inline float ABS2(const Complex &a)
{
    float result = a.dat[0] * a.dat[0] + a.dat[1] * a.dat[1];
    return result;
};

inline Complex COMPLEX(float a, float b)
{
    Complex z;
    z.dat[0] = a;
    z.dat[1] = b;
    return z;
};
inline float REAL(const Complex& a)
{
    return a.dat[0];
};
inline float IMAG(const Complex& a)
{
    return a.dat[1];
};

inline float gsl_real(const Complex& a)
{
    return a.dat[0];
};

inline float gsl_imag(const Complex& a)
{
    return a.dat[1];
};

inline float gsl_real_imag_sum(const Complex& a)
{
    return a.dat[0] + a.dat[1];
};

inline Complex operator-(const Complex& a)
{
    //return COMPLEX(-REAL(a), -IMAG(a));
    Complex newa;
    newa.dat[0] =  0.0 - a.dat[0];
    newa.dat[1] =  0.0 - a.dat[1];
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
    //return gsl_complex_sub(a, b);
    Complex result;
    result.dat[0] = a.dat[0] - b.dat[0];
    result.dat[1] = a.dat[1] - b.dat[1];
    return result;

};

inline Complex operator*(const Complex& a, const Complex& b)
{
    //return gsl_complex_mul(a, b);
    Complex result;
    result.dat[0] = a.dat[0] * b.dat[0];
    result.dat[1] = a.dat[1] * b.dat[1];
    return result;

};

inline Complex operator/(const Complex& a, const Complex& b)
{
    //return gsl_complex_div(a, b);
    Complex result;
    result.dat[0] = a.dat[0] / b.dat[0];
    result.dat[1] = a.dat[1] / b.dat[1];
    return result;

};

inline void operator+=(Complex& a, const Complex b) { a = a + b; };

inline void operator-=(Complex& a, const Complex b) { a = a - b; };

inline void operator*=(Complex& a, const Complex b) { a = a * b; };

inline void operator/=(Complex& a, const Complex b) { a = a / b; };

inline Complex operator*(const Complex a, const float x)
{
    /*
     *return gsl_complex_mul_real(a, x);
     */
    Complex result;
    result.dat[0] = a.dat[0] * x;
    result.dat[1] = a.dat[1] * x;
    return result;

};

inline Complex operator*(const float x, const Complex a)
{
    /*
     *return a * x;
     */
    Complex result;
    result.dat[0] = a.dat[0] * x;
    result.dat[1] = a.dat[1] * x;
    return result;

};

inline void operator*=(Complex& a, const float x)
{
    /*
     *a = a * x;
     */
    a.dat[0] = a.dat[0] * x;
    a.dat[1] = a.dat[1] * x;
};

inline Complex operator/(const Complex a, const float x)
{
    /*
     *return gsl_complex_div_real(a, x);
     */
    Complex result;
    result.dat[0] = a.dat[0] / x;
    result.dat[1] = a.dat[1] / x;
    return result;

};

inline void operator/=(Complex& a, const float x)
{
    /*
     *a = a / x; 
     */
    a.dat[0] = a.dat[0] / x;
    a.dat[1] = a.dat[1] / x;

};

#endif // COMPLEX_H
