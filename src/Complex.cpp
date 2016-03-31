/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Complex.h"

Complex operator-(Complex a)
{
    return COMPLEX(-GSL_REAL(a), -GSL_IMAG(a));
}

Complex operator+(Complex a, Complex b)
{
    return gsl_complex_add(a, b);
}

Complex operator-(Complex a, Complex b)
{
    return gsl_complex_sub(a, b);
}

Complex operator*(Complex a, Complex b)
{
    return gsl_complex_mul(a, b);
}

Complex operator/(Complex a, Complex b)
{
    return gsl_complex_div(a, b);
}

void operator+=(Complex& a, Complex b) { a = a + b; }

void operator-=(Complex& a, Complex b) { a = a - b; }

void operator*=(Complex& a, Complex b) { a = a * b; }

void operator/=(Complex& a, Complex b) { a = a / b; }

Complex operator*(Complex a, double x)
{
    return gsl_complex_mul_real(a, x);
}

Complex operator*(double x, Complex a)
{
    return a * x;
}

void operator*=(Complex& a, double x) { a = a * x; }

Complex operator/(Complex a, double x)
{
    return gsl_complex_div_real(a, x);
}

void operator/=(Complex a, double x) { a = a / x; }
