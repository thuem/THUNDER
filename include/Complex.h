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

Complex operator-(Complex a);

Complex operator+(Complex a, Complex b);

Complex operator-(Complex a, Complex b);

Complex operator*(Complex a, Complex b);

Complex operator/(Complex a, Complex b);

void operator+=(Complex& a, Complex b);

void operator-=(Complex& a, Complex b);

void operator*=(Complex& a, Complex b);

void operator/=(Complex& a, Complex b);

Complex operator*(Complex a, double x);

Complex operator*(double x, Complex a);

void operator*=(Complex& a, double x);

Complex operator/(Complex a, double x);

void operator/=(Complex a, double x);

#endif // COMPLEX_H
