/** @file
 *  @brief Complex.h defines complex number related operations like +,-,*,/,|a|,@f$|a|^2@f$ and so on.
 */
#ifndef COMPLEX_H
#define COMPLEX_H

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <math.h>

#include "Config.h"
#include "Precision.h"
#include "Typedef.h"


inline Complex COMPLEX_POLAR(const RFLOAT phi)
{
    Complex z;

    z.dat[0] = TS_COS(phi);
    z.dat[1] = TS_SIN(phi);

    return z;
}



inline Complex CONJUGATE(const Complex &a)
{
    Complex z;
    z.dat[0] = a.dat[0];
    z.dat[1] = -a.dat[1];
    return z;
}

static RFLOAT ts_hypot(const RFLOAT x,
                       const RFLOAT y)
{
#ifdef SINGLE_PRECISION
    RFLOAT xabs = fabsf(x) ;
    RFLOAT yabs = fabsf(y) ;
#else
    RFLOAT xabs = fabs(x) ;
    RFLOAT yabs = fabs(y) ;
#endif

    RFLOAT min, max;

    if (xabs < yabs)
    {
        min = xabs;
        max = yabs;
    }
    else
    {
        min = yabs;
        max = xabs;
    }

    if (min == 0) 
    {
        return max;
    }

    RFLOAT u = min / max;

    return max * TS_SQRT(1 + u * u);
};

inline RFLOAT ABS(const Complex& a)
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
    Complex newa;

    newa.dat[0] = -a.dat[0];
    newa.dat[1] = -a.dat[1];

    return newa;
};

inline Complex operator+(const Complex& a, const Complex& b)
{
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

    result.dat[0] = a.dat[0] * b.dat[0] - a.dat[1] * b.dat[1];
    result.dat[1] = a.dat[0] * b.dat[1] + a.dat[1] * b.dat[0];

    return result;
};

inline Complex operator/(const Complex& a, const Complex& b)
{
    Complex result;

    result.dat[0] = a.dat[0] * b.dat[0] + a.dat[1] * b.dat[1];
    result.dat[1] = a.dat[1] * b.dat[0] - a.dat[0] * b.dat[1]; 
    
    RFLOAT norm = b.dat[0] * b.dat[0] + b.dat[1] * b.dat[1];

    result.dat[0] /= norm;
    result.dat[1] /= norm;

    return result;
};

inline void operator+=(Complex& a, const Complex b) { a = a + b; };

inline void operator-=(Complex& a, const Complex b) { a = a - b; };

inline void operator*=(Complex& a, const Complex b) { a = a * b; };

inline void operator/=(Complex& a, const Complex b) { a = a / b; };

inline Complex operator*(const Complex a, const RFLOAT x)
{
    Complex result;

    result.dat[0] = a.dat[0] * x;
    result.dat[1] = a.dat[1] * x;

    return result;
};

inline Complex operator*(const RFLOAT x, const Complex a)
{
    Complex result;

    result.dat[0] = a.dat[0] * x;
    result.dat[1] = a.dat[1] * x;

    return result;
};

inline void operator*=(Complex& a, const RFLOAT x)
{
    a.dat[0] = a.dat[0] * x;
    a.dat[1] = a.dat[1] * x;
};

inline Complex operator/(const Complex a, const RFLOAT x)
{
    Complex result;

    result.dat[0] = a.dat[0] / x;
    result.dat[1] = a.dat[1] / x;

    return result;

};

inline void operator/=(Complex& a, const RFLOAT x)
{
    a.dat[0] = a.dat[0] / x;
    a.dat[1] = a.dat[1] / x;
};

#endif // COMPLEX_H
