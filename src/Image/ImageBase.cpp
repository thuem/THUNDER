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

ImageBase::~ImageBase() {}

const double& ImageBase::iGetRL(size_t i) const
{
    return _dataRL[i];
};

const Complex& ImageBase::iGetFT(size_t i) const
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
    return !_dataRL;
}

bool ImageBase::isEmptyFT() const
{
    return !_dataFT;
}

size_t ImageBase::sizeRL() const { return _sizeRL; }

size_t ImageBase::sizeFT() const { return _sizeFT; }

void ImageBase::mtxIniRL()
{
    _mtxRL.reset(new mutex[_sizeRL]);
}

void ImageBase::mtxIniFT()
{
    _mtxRL.reset(new mutex[_sizeFT]);
}

void ImageBase::mtxClrRL()
{
    _mtxRL.reset();
}

void ImageBase::mtxClrFT()
{
    _mtxFT.reset();
}

void ImageBase::clear()
{
    clearRL();
    clearFT();
}

void ImageBase::clearRL()
{
    _dataRL.reset();
    _sizeRL = 0;
}

void ImageBase::clearFT()
{
    _dataFT.reset();
    _sizeFT = 0;
}

double norm(ImageBase& base)
{
    return sqrt(cblas_dznrm2(base.sizeFT(), &base[0], 1));
}

void normalise(ImageBase& base)
{
    double mean = gsl_stats_mean(&base(0), 1, base.sizeRL());
    double stddev = gsl_stats_sd_m(&base(0), 1, base.sizeRL(), mean);

    FOR_EACH_PIXEL_RL(base)
        base(i) -= mean;

    SCALE_RL(base, 1.0 / stddev);
}

void ImageBase::copyBase(ImageBase& other) const
{
    other._sizeRL = _sizeRL;
    if (_dataRL)
    {
        other._dataRL.reset(new double[_sizeRL]);
        memcpy(other._dataRL.get(), _dataRL.get(), _sizeRL * sizeof(_dataRL[0]));
    }
    else
        other._dataRL.reset();

    other._sizeFT = _sizeFT;
    if (_dataFT)
    {
        other._dataFT.reset(new Complex[_sizeFT]);
        memcpy(other._dataFT.get(), _dataFT.get(), _sizeFT * sizeof(_dataFT[0]));
    }
    else
        other._dataFT.reset();
}
