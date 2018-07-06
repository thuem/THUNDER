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

#include "Config.h"
#include "Precision.h"

#include "Device.cuh"

namespace cuthunder {

class Complex
{
    public:

        HD_CALLABLE Complex(RFLOAT r = 0, RFLOAT i = 0)
                    : _real(r), _imag(i) {}

        HD_CALLABLE ~Complex() {}
        
        HD_CALLABLE Complex(const Complex& that)
                    : _real(that._real), _imag(that._imag) {}

        HD_CALLABLE __forceinline__ RFLOAT real() const { return _real; }
        
        HD_CALLABLE __forceinline__ RFLOAT imag() const { return _imag; }

        HD_CALLABLE __forceinline__ RFLOAT* realAddr() { return &_real; }
        
        HD_CALLABLE __forceinline__ RFLOAT* imagAddr() { return &_imag; }

        HD_CALLABLE __forceinline__ void set(const RFLOAT real, const RFLOAT imag) {
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

        HD_CALLABLE __forceinline__ Complex& operator*=(const RFLOAT scale) {
            _real *= scale;
            _imag *= scale;
            return *this;
        }

        HD_CALLABLE __forceinline__ Complex& operator*=(const Complex& that) {
            RFLOAT real = _real * that._real - _imag * that._imag;
            RFLOAT imag = _real * that._imag + _imag * that._real;

            _real = real;
            _imag = imag;
            return *this;
        }
        
        HD_CALLABLE __forceinline__ Complex operator*(const RFLOAT scale) const {
            return Complex(_real * scale, _imag * scale);
        }

        HD_CALLABLE __forceinline__ Complex operator*(const Complex& that) const {
            RFLOAT real = _real * that._real - _imag * that._imag;
            RFLOAT imag = _real * that._imag + _imag * that._real;
            return Complex(real, imag);
        }
    
    
    private:

        RFLOAT _real;
        RFLOAT _imag;
        
};

}

#endif
