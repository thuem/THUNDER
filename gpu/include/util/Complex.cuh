/***********************************************************************
 * FileName: Complex.cuh
 * Author  : Kunpeng WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#ifndef COMPLEX_CUH
#define COMPLEX_CUH

#include "Device.cuh"

namespace cuthunder {

class Complex
{
    public:

        HD_CALLABLE Complex(double r = 0, double i = 0)
                    : _real(r), _imag(i) {}

        HD_CALLABLE ~Complex() {}
        
        HD_CALLABLE Complex(const Complex& that)
                    : _real(that._real), _imag(that._imag) {}

        HD_CALLABLE __forceinline__ double real() const { return _real; }
        
        HD_CALLABLE __forceinline__ double imag() const { return _imag; }

        HD_CALLABLE __forceinline__ double* realAddr() { return &_real; }
        
        HD_CALLABLE __forceinline__ double* imagAddr() { return &_imag; }

        HD_CALLABLE __forceinline__ void set(const double real, const double imag) {
        	_real = real; _imag = imag;
        }

        HD_CALLABLE __forceinline__ Complex& conj() {
            _imag *= -1;
            return *this;
        }

        HD_CALLABLE __forceinline__ Complex& operator+=(const Complex& that) {
            _real += that._real;
            _imag += that._imag;
            return *this;
        }
        
        HD_CALLABLE __forceinline__ Complex operator+(const Complex& that) const {
            return Complex(_real + that._real, _imag + that._imag);
        }

        HD_CALLABLE __forceinline__ Complex operator-(const Complex& that) const {
            return Complex(_real - that._real, _imag - that._imag);
        }

        HD_CALLABLE __forceinline__ Complex& operator*=(const double scale) {
            _real *= scale;
            _imag *= scale;
            return *this;
        }

        HD_CALLABLE __forceinline__ Complex& operator*=(const Complex& that) {
            double real = _real * that._real - _imag * that._imag;
            double imag = _real * that._imag + _imag * that._real;

            _real = real;
            _imag = imag;
            return *this;
        }
        
        HD_CALLABLE __forceinline__ Complex operator*(const double scale) const {
            return Complex(_real * scale, _imag * scale);
        }

        HD_CALLABLE __forceinline__ Complex operator*(const Complex& that) const {
            double real = _real * that._real - _imag * that._imag;
            double imag = _real * that._imag + _imag * that._real;
            return Complex(real, imag);
        }
    
    private:

        double _real;
        double _imag;
        
};

}

#endif
