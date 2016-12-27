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

#include "Typedef.h"

#define CONJUGATE(a) gsl_complex_conjugate(a)

#define ABS(a) gsl_complex_abs(a)

#define ABS2(a) gsl_complex_abs2(a)

#define COMPLEX_POLAR(phi) gsl_complex_polar(1, phi)

#define COMPLEX(a, b) gsl_complex_rect(a, b)

#define REAL(a) GSL_REAL(a)

#define IMAG(a) GSL_IMAG(a)

inline double gsl_real(const Complex a)
{
    return REAL(a);
};

inline double gsl_imag(const Complex a)
{
    return IMAG(a);
};

inline double gsl_real_imag_sum(const Complex a)
{
    return REAL(a) + IMAG(a);
};

inline Complex operator-(const Complex a)
{
    return COMPLEX(-REAL(a), -IMAG(a));
};

inline Complex operator+(const Complex a, const Complex b)
{
    return gsl_complex_add(a, b);
};

inline Complex operator-(const Complex a, const Complex b)
{
    return gsl_complex_sub(a, b);
};

inline Complex operator*(const Complex a, const Complex b)
{
    return gsl_complex_mul(a, b);
};

inline Complex operator/(const Complex a, const Complex b)
{
    return gsl_complex_div(a, b);
};

inline void operator+=(Complex& a, const Complex b) { a = a + b; };

inline void operator-=(Complex& a, const Complex b) { a = a - b; };

inline void operator*=(Complex& a, const Complex b) { a = a * b; };

inline void operator/=(Complex& a, const Complex b) { a = a / b; };

inline Complex operator*(const Complex a, const double x)
{
    return gsl_complex_mul_real(a, x);
};

inline Complex operator*(const double x, const Complex a)
{
    return a * x;
};

inline void operator*=(Complex& a, const double x) { a = a * x; };

inline Complex operator/(const Complex a, const double x)
{
    return gsl_complex_div_real(a, x);
};

inline void operator/=(Complex& a, const double x) { a = a / x; };

#endif // COMPLEX_H
