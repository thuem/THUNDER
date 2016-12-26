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

ImageBase::~ImageBase()
{
#ifdef FFTW_PTR
    if (_dataRL != NULL)
    {
        delete[] _dataRL;
        _dataRL = NULL;
    }

    if (_dataFT != NULL)
    {
        delete[] _dataFT;
        _dataFT = NULL;
    }
#endif
}

/***
const double& ImageBase::iGetRL(const size_t i) const
{
    return _dataRL[i];
};

const Complex& ImageBase::iGetFT(const size_t i) const
{
    return _dataFT[i];
};
***/

/***
double& ImageBase::operator()(const size_t i)
{
    return _dataRL[i];
}

Complex& ImageBase::operator[](const size_t i)
{
    return _dataFT[i];
}
***/

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

void ImageBase::clear()
{
    clearRL();
    clearFT();
}

void ImageBase::clearRL()
{
#ifdef CXX11_PTR
    _dataRL.reset();
#endif
    
#ifdef FFTW_PTR
    fftw_free(_dataRL);

    _dataRL = NULL;
#endif
}

void ImageBase::clearFT()
{
#ifdef CXX11_PTR
    _dataFT.reset();
#endif
    
#ifdef FFTW_PTR
    fftw_free(_dataFT);

    _dataFT = NULL;
#endif
}

void ImageBase::copyBase(ImageBase& other) const
{
    other._sizeRL = _sizeRL;

    if (_dataRL)
    {
#ifdef CXX11_PTR
        other._dataRL.reset(new double[_sizeRL]);

        memcpy(other._dataRL.get(), _dataRL.get(), _sizeRL * sizeof(_dataRL[0]));
#endif

#ifdef FFTW_PTR
        //other._dataRL = fftw_alloc_real(_sizeRL);
        other._dataRL = (double*)fftw_malloc(_sizeRL * sizeof(double));

        memcpy(other._dataRL, _dataRL, _sizeRL * sizeof(double));
#endif
    }
    else
    {
#ifdef CXX11_PTR
        other._dataRL.reset();
#endif

#ifdef FFTW_PTR
        other._dataRL = NULL;
#endif
    }

    other._sizeFT = _sizeFT;

    if (_dataFT)
    {
#ifdef CXX11_PTR
        other._dataFT.reset(new Complex[_sizeFT]);

        memcpy(other._dataFT.get(), _dataFT.get(), _sizeFT * sizeof(_dataFT[0]));
#endif

#ifdef FFTW_PTR
        //other._dataFT = (Complex*)fftw_alloc_complex(_sizeFT);
        other._dataFT = (Complex*)fftw_malloc(_sizeFT * sizeof(Complex));
        
        memcpy(other._dataFT, _dataFT, _sizeFT * sizeof(Complex));
#endif
    }
    else
    {
#ifdef CXX11_PTR
        other._dataFT.reset();
#endif

#ifdef FFTW_PTR
        other._dataFT = NULL;
#endif
    }
}

ImageBase ImageBase::copyBase() const
{
    ImageBase that;

    copyBase(that);

    return that;
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
