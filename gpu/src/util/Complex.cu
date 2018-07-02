/***********************************************************************
 * FileName: Complex.cu
 * Author  : Kunpeng WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#include "Complex.cuh"

namespace cuthunder {

//HD_CALLABLE void Complex::set(const double real,
//	                          const double imag)
//{
//	_real = real; _imag = imag;
//}
//
//HD_CALLABLE Complex& Complex::conj()
//{
//    _imag *= -1;
//    return *this;
//}
//
//HD_CALLABLE Complex& Complex::operator+=(const Complex& that)
//{
//    _real += that._real;
//    _imag += that._imag;
//
//    return *this;
//}
//
//HD_CALLABLE Complex Complex::operator+(const Complex& that) const
//{
//    return Complex(_real + that._real, _imag + that._imag);
//}
//
//HD_CALLABLE Complex& Complex::operator*=(const double scale)
//{
//    _real *= scale;
//    _imag *= scale;
//
//    return *this;
//}
//
//HD_CALLABLE Complex& Complex::operator*=(const Complex& that)
//{
//    double real = _real * that._real - _imag * that._imag;
//    double imag = _real * that._imag + _imag * that._real;
//
//    _real = real;
//    _imag = imag;
//
//    return *this;
//}
//
//HD_CALLABLE Complex Complex::operator*(const double scale) const
//{
//    return Complex(_real * scale, _imag * scale);
//}
//
//HD_CALLABLE Complex Complex::operator*(const Complex& that) const
//{
//    double real = _real * that._real - _imag * that._imag;
//    double imag = _real * that._imag + _imag * that._real;
//
//    return Complex(real, imag);
//}

} // end namespace cuthunder
