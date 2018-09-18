/** @file
 *  @author Huabin Ruan 
 *  @author Mingxu Hu 
 *  @version 1.4.11.080913
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Huabin Ruan | 2018/09/13 | 1.4.11.080913 | add header for file and functions
 *  Huabin Ruan | 2018/09/18 | 1.4.11.080918 | move arithmetic expression into latex tag 
 *
 *  @brief Complex.h contains functions used for complex number related operations, like +, -, *, /, |c|.
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
 *  Get the complex number representation @f$\mathbf{c}@f$, given the angle @f$\phi@f$ in polar coordinate.
 *  @return Complex polar representation.
 */
inline Complex COMPLEX_POLAR(const RFLOAT phi /**< [in] @f$\phi@f$ - Angle value @f$\phi@f$ to be converted */)
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
 *  @brief Calculate the @f$|a|@f$ of complex number @f$a@f$.
 *
 *  @return The @f$|a|@f$ of complex number @f$a@f$.
 */
inline RFLOAT ABS(const Complex &a /**< [in] The number whose @f$|a|@f$ needs to be calculated*/)
{
    return ts_hypot(a.dat[0], a.dat[1]);
}

/**
 *  @brief Calculate the @f$|a|^2@f$ of complex number @f$a@f$.
 *
 *  @return The @f$|a|^2@f$ of complex number @f$a@f$.
 */
inline RFLOAT ABS2(const Complex &a /**< [in] The number whose @f$|a|^2@f$ needs to be calculated  */)
{
    RFLOAT result = a.dat[0] * a.dat[0] + a.dat[1] * a.dat[1];
    return result;
}


/**
 *  @brief Construct a complex number with @f$a@f$ as real part and @f$b@f$ as image part.
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
 *  @brief Get the real part of the complex number @f$a@f$
 *
 *  @return Complex number @f$a'@f$s real part
 */
inline RFLOAT REAL(const Complex &a /**< [in] Complex number to be operated */)
{
    return a.dat[0];
}


/**
 *  @brief Get the image part of the complex number @f$a@f$
 *
 *  @return Complex number @f$a'@f$s image part
 */
inline RFLOAT IMAG(const Complex &a /**< [in] Complex number to be operated */)
{
    return a.dat[1];
}


/**
 *  @brief Calcuate the sum of the complex number @f$a'@f$s real part and image part
 *
 *  @return The sum of @f$a'@f$s real part and image part
 */
inline RFLOAT gsl_real_imag_sum(const Complex &a /**< [in] Complex number to be operated */)
{
    return a.dat[0] + a.dat[1];
}

/**
 *  @brief Implement the add operation between two complex numbers, e.g. @f$c = a + b@f$, where @f$a@f$ and @f$b@f$ are complex numbers.
 *
 *  @return the result of @f$c = a + b@f$.
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
 *  @brief Implement the sub operation between two complex numbers, e.g. @f$c = a - b@f$, where @f$a@f$ and @f$b@f$ are complex numbers.
 *
 *  @return the result of @f$c = a - b@f$.
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
 *  @brief Implement the mul operation between two complex numbers, e.g. @f$c = a * b@f$, where @f$a@f$ and @f$b@f$ are complex numbers.
 *
 *  @return the result of @f$c = a * b@f$.
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
 *  @brief Implement the div operation between two complex numbers, e.g. @f$c = a / b@f$, where @f$a@f$ and @f$b@f$ are complex numbers.
 *
 *  @return the result of @f$c = a / b@f$.
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
 *  @brief Implement the @f$+=@f$ operation between two complex numbers, e.g. @f$a += b@f$, where @f$a@f$ and @f$b@f$ are complex numbers.
 *
 *  @return the result of @f$a += b@f$.
 */
inline void operator+=(Complex &a,       /**< [in] First operand used to perform @f$+=@f$ operation between two complex numbers */
                       const Complex &b /**< [in] Second operand used to perform @f$+=@f$ operation between two complex numbers */
                      )
{
    a = a + b;
}

/**
 *  @brief Implement the @f$-=@f$ operation between two complex numbers, e.g. @f$a -= b@f$, where @f$a@f$ and @f$b@f$ are complex numbers.
 *
 *  @return the result of @f$a -= b@f$.
 */
inline void operator-=(Complex &a,       /**< [in] First operand used to perform @f$-=@f$ operation between two complex numbers */
                       const Complex &b /**< [in] Second operand used to perform @f$-=@f$ operation between two complex numbers */
                      )
{
    a = a - b;
}

/**
 *  @brief Implement the *= operation between two complex numbers, e.g. @f$a *= b@f$, where @f$a@f$ and @f$b@f$ are complex numbers.
 *
 *  @return the result of @f$a *= b@f$.
 */
inline void operator*=(Complex &a,       /**< [in] First operand used to perform @f$*=@f$ operation between two complex numbers */
                       const Complex &b /**< [in] Second operand used to perform @f$*=@f$ operation between two complex numbers */
                      )
{
    a = a * b;
}

/**
 *  @brief Implement the @f$/=@f$ operation between two complex numbers, e.g. @f$a /= b@f$, where @f$a@f$ and @f$b@f$ are complex numbers.
 *
 *  @return the result of @f$a /= b@f$.
 */
inline void operator/=(Complex &a,      /**< [in] First operand used to perform @f$/=@f$ operation between two complex numbers */
                       const Complex &b /**< [in] Second operand used to perform @f$/=@f$ operation between two complex numbers */
                      )
{
    a = a / b;
}

/**
 *  @brief Implement the mul operation between a  complex number and a RFLOAT number, e.g. @f$c = a * x@f$, where @f$a@f$is @f$a@f$ complex number and @f$x@f$ is a RFLOAT number.
 *
 *  @return the result of @f$c = a * x@f$.
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

/**
 *  @brief Implement the mul operation between a  RFLOAT number and a Complex number, e.g. @f$c = x * a@f$, where @f$x @f$is a RFLOAT number and @f$a@f$is a complex number.
 *
 *  @return the result of @f$c = x * a@f$.
 */
inline Complex operator*(const RFLOAT x, /**< [in] First operand with type of RFLOAT used to perform mul operation.*/
                         const Complex a /**< [in] Second operand with type of complex used to perform mul operation.*/
                        )
{
    Complex result;
    result.dat[0] = a.dat[0] * x;
    result.dat[1] = a.dat[1] * x;
    return result;
}

/**
 *  @brief Implement the mul operation between a  complex number and a RFLOAT number, e.g. @f$a *=  x@f$, where @f$a@f$ is a complex number and @f$x@f$ is a RFLOAT number.
 *
 *  @return the result of @f$a *= x@f$.
 */
inline void operator*=(Complex& a,    /**< [in] First operand with type of complex used to perform @f$*=@f$ operation.*/
                       const RFLOAT x /**< [in] Second operand with type of RFLOAT used to perform @f$*=@f$ operation. */
                      )
{
    a.dat[0] = a.dat[0] * x;
    a.dat[1] = a.dat[1] * x;
}

/**
 *  @brief Implement the div operation between a  complex number and a RFLOAT number, e.g. @f$c = a / x@f$, where @f$a@f$ is a complex number and @f$x@f$ is a RFLOAT number.
 *
 *  @return the result of @f$c = a / x@f$.
 */
inline Complex operator/(const Complex a, /**< [in] First operand with type of complex used to perform div operation.*/
                         const RFLOAT x   /**< [in] Second operand with type of RFLOAT used to perform div operation. */
                        )
{
    Complex result;
    result.dat[0] = a.dat[0] / x;
    result.dat[1] = a.dat[1] / x;
    return result;
}

/**
 *  @brief Implement the @f$/=@f$ operation between a  complex number and a RFLOAT number, e.g. @f$a /=  x@f$, where @f$a@f$ is a complex number and @f$x@f$ is a RFLOAT number.
 *
 *  @return the result of @f$a /= x@f$.
 */
inline void operator/=(Complex& a,    /**< [in] First operand with type of complex used to perform @f$/=@f$ operation.*/
                       const RFLOAT x /**< [in] Second operand with type of RFLOAT used to perform @f$/=@f$ operation. */
                      )
{
    a.dat[0] = a.dat[0] / x;
    a.dat[1] = a.dat[1] / x;
}

#endif // COMPLEX_H
