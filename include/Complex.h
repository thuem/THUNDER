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


/**
 *  @brief Get the polar representation based on angle value @f$\phi@f$.
 *
 *  @return Complex polar representation.
 */
inline Complex COMPLEX_POLAR(const RFLOAT phi /**< [in] Angle value @f$\phi@f$ */)
{
    Complex z;
    z.dat[0] = TS_COS(phi);
    z.dat[1] = TS_SIN(phi);
    return z;
}



/**
 * @brief Get the conjugate result based on value a
 *
 * @return Conjugate of a.
 */
inline Complex CONJUGATE(const Complex &a /**< [in] Complex number whose conjuate value needs to be returned */)
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
}


/**
 *  @brief Calculate the |a| of complex number a.
 *
 *  @return The |a| of complex number a.
 */
inline RFLOAT ABS(const Complex &a /**< [in]  The number whose |a| needs to be calculated*/)
{
    return ts_hypot(a.dat[0], a.dat[1]);
}

inline RFLOAT ABS2(const Complex &a)
{
    RFLOAT result = a.dat[0] * a.dat[0] + a.dat[1] * a.dat[1];
    return result;
}


/**
 *  @brief Construct a complex number with a as real part and b as image part.
 *
 *  @return A initialized complex number.
 */
inline Complex COMPLEX(RFLOAT a, /**< [in] Real part */
                       RFLOAT b  /**< [in] Image part */
                      )
{
    Complex z;
    z.dat[0] = a;
    z.dat[1] = b;
    return z;
}


/**
 *  @brief Get the real part of the complex number a
 *
 *  @return Complex number a's real part
 */
inline RFLOAT REAL(const Complex &a /**< [in] Complex number to be operated */)
{
    return a.dat[0];
}


/**
 *  @brief Get the image part of the complex number a
 *
 *  @return Complex number a's image part
 */
inline RFLOAT IMAG(const Complex &a /**< [in] Complex number to be operated */)
{
    return a.dat[1];
}


/**
 *  @brief Calcuate the sum of the complex number a's real part and image part
 *
 *  @return The sum of a's real part and image part
 */
inline RFLOAT gsl_real_imag_sum(const Complex &a /**< [in] Complex number to be operated */)
{
    return a.dat[0] + a.dat[1];
}

/**
 *  @brief Implement the add operation between two complex numbers, e.g. c = a + b, where a and b are complex numbers.
 *
 *  @return the result of c = a + b.
 */
inline Complex operator+(const Complex &a, /**< [in] First operand used to perform add operation between two complex numbers */
                         const Complex &b  /**< [in] Second operand used to perform add operation between two complex numbers */
                        )
{
    Complex result;
    result.dat[0] = a.dat[0] + b.dat[0];
    result.dat[1] = a.dat[1] + b.dat[1];
    return result;
}

/**
 *  @brief Implement the sub operation between two complex numbers, e.g. c = a - b, where a and b are complex numbers.
 *
 *  @return the result of c = a - b.
 */
inline Complex operator-(const Complex &a, /**< [in] First operand used to perform sub operation between two complex numbers */
                         const Complex &b  /**< [in] Second operand used to perform sub operation between two complex numbers */
                        )
{
    Complex result;
    result.dat[0] = a.dat[0] - b.dat[0];
    result.dat[1] = a.dat[1] - b.dat[1];
    return result;
}

/**
 *  @brief Implement the mul operation between two complex numbers, e.g. c = a * b, where a and b are complex numbers.
 *
 *  @return the result of c = a * b.
 */
inline Complex operator*(const Complex &a, /**< [in] First operand used to perform sub operation between two complex numbers */
                         const Complex &b  /**< [in] Second operand used to perform sub operation between two complex numbers */
                        )
{
    Complex result;
    result.dat[0] = a.dat[0] * b.dat[0] - a.dat[1] * b.dat[1];
    result.dat[1] = a.dat[0] * b.dat[1] + a.dat[1] * b.dat[0];
    return result;
}

/**
 *  @brief Implement the div operation between two complex numbers, e.g. c = a / b, where a and b are complex numbers.
 *
 *  @return the result of c = a / b.
 */
inline Complex operator/(const Complex &a, /**< [in] First operand used to perform div operation between two complex numbers */
                         const Complex &b  /**< [in] Second operand used to perform div operation between two complex numbers */
                        )
{
    Complex result;
    result.dat[0] = a.dat[0] * b.dat[0] + a.dat[1] * b.dat[1];
    result.dat[1] = a.dat[1] * b.dat[0] - a.dat[0] * b.dat[1];
    RFLOAT norm = b.dat[0] * b.dat[0] + b.dat[1] * b.dat[1];
    result.dat[0] /= norm;
    result.dat[1] /= norm;
    return result;
}

/**
 *  @brief Implement the += operation between two complex numbers, e.g. a += b, where a and b are complex numbers.
 *
 *  @return the result of a += b.
 */
inline void operator+=(Complex &a,       /**< [in] First operand used to perform += operation between two complex numbers */
                       const Complex &b /**< [in] Second operand used to perform += operation between two complex numbers */
                      )
{
    a = a + b;
}

/**
 *  @brief Implement the -= operation between two complex numbers, e.g. a -= b, where a and b are complex numbers.
 *
 *  @return the result of a -= b.
 */
inline void operator-=(Complex &a,       /**< [in] First operand used to perform -= operation between two complex numbers */
                       const Complex &b /**< [in] Second operand used to perform -= operation between two complex numbers */
                      )
{
    a = a - b;
}

/**
 *  @brief Implement the *= operation between two complex numbers, e.g. a *= b, where a and b are complex numbers.
 *
 *  @return the result of a *= b.
 */
inline void operator*=(Complex &a,       /**< [in] First operand used to perform *= operation between two complex numbers */
                       const Complex &b /**< [in] Second operand used to perform *= operation between two complex numbers */
                      )
{
    a = a * b;
}

/**
 *  @brief Implement the /= operation between two complex numbers, e.g. a /= b, where a and b are complex numbers.
 *
 *  @return the result of a /= b.
 */
inline void operator/=(Complex &a,       /**< [in] First operand used to perform /= operation between two complex numbers */
                       const Complex &b /**< [in] Second operand used to perform /= operation between two complex numbers */
                      )
{
    a = a / b;
}

/**
 *  @brief Implement the mul operation between a  complex number and a RFLOAT number, e.g. c = a + b, where a is a complex number and b is a RFLOAT number.
 *
 *  @return the result of c = a * b.
 */
inline Complex operator*(const Complex a, /**< [in] First operand with type of complex used to perform mul operation.*/
                         const RFLOAT x   /**< [in] Second operand with type of RFLOAT used to perform mul operation. */
                        )
{
    Complex result;
    result.dat[0] = a.dat[0] * x;
    result.dat[1] = a.dat[1] * x;
    return result;
}

inline Complex operator*(const RFLOAT x, const Complex a)
{
    Complex result;
    result.dat[0] = a.dat[0] * x;
    result.dat[1] = a.dat[1] * x;
    return result;
}

inline void operator*=(Complex &a, const RFLOAT x)
{
    a.dat[0] = a.dat[0] * x;
    a.dat[1] = a.dat[1] * x;
}

inline Complex operator/(const Complex a, const RFLOAT x)
{
    Complex result;
    result.dat[0] = a.dat[0] / x;
    result.dat[1] = a.dat[1] / x;
    return result;
}

inline void operator/=(Complex &a, const RFLOAT x)
{
    a.dat[0] = a.dat[0] / x;
    a.dat[1] = a.dat[1] / x;
}

#endif // COMPLEX_H
