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

#define COMPLEX(a, b) gsl_complex_rect(a, b)

Complex operator-(Complex a);

Complex operator+(Complex a, Complex b);

Complex operator-(Complex a, Complex b);

Complex operator*(Complex a, Complex b);

Complex operator/(Complex a, Complex b);

void operator+=(Complex a, Complex b);

void operator-=(Complex a, Complex b);

void operator*=(Complex a, Complex b);

void operator/=(Complex a, Complex b);

Complex operator*(Complex a, double x);

Complex operator*(double x, Complex a);

void operator*=(Complex a, double x);

#endif // COMPLEX_H
