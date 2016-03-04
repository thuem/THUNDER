/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "ImageBase.h"

ImageBase::ImageBase() {}

ImageBase::ImageBase(const ImageBase& that)
{
    *this = that;
}

ImageBase& ImageBase::operator=(const ImageBase& that)
{
    clear();

    _sizeRL = that.sizeRL();
    _sizeFT = that.sizeFT();

    if (!that.isEmptyRL())
    {
        _dataRL = new double [_sizeRL];
        memcpy(_dataRL, &that.getRL(), sizeof(double) * _sizeRL);
    }

    if (!that.isEmptyFT())
    {
        _dataFT = new Complex[_sizeFT];
        memcpy(_dataFT, &that.getFT(), sizeof(Complex) * _sizeFT);
    }
}

const double& ImageBase::getRL(size_t i) const
{
    return _dataRL[i];
};

const Complex& ImageBase::getFT(size_t i) const
{
    return _dataFT[i];
};

double& ImageBase::operator()(const size_t i)
{
    return _dataRL[i];
}

Complex& ImageBase::operator[](const size_t i)
{
    return _dataFT[i];
}

bool ImageBase::isEmptyRL() const
{
    return (_dataRL == NULL);
}

bool ImageBase::isEmptyFT() const
{
    return (_dataFT == NULL);
}

size_t ImageBase::sizeRL() const { return _sizeRL; }

size_t ImageBase::sizeFT() const { return _sizeFT; }

void ImageBase::clear()
{
    clearRL();
    clearFT();
}

void ImageBase::clearRL()
{
    if (_dataRL != NULL)
    {
        delete[] _dataRL;
        _dataRL = NULL;
        _sizeRL = 0;
    }
}

void ImageBase::clearFT()
{
    if (_dataFT != NULL)
    {
        delete[] _dataFT;
        _dataFT = NULL;
        _sizeFT = 0;
    }
}

double norm(ImageBase& base)
{
    return sqrt(cblas_dznrm2(base.sizeFT(), &base[0], 1));
}

void normalise(ImageBase& base)
{
    /***
    gsl_vector vec;
    vec.size = base.sizeRL();
    vec.data = &base(0);

    normalise(vec);
    ***/
}
