/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Volume.h"

Volume::Volume() : _nCol(0), _nRow(0), _nSlc(0) {}

Volume::Volume(const int nCol,
               const int nRow,
               const int nSlc,
               const int space)
{
    alloc(nCol, nRow, nSlc, space);
}

Volume::~Volume() {}

void Volume::swap(Volume& that)
{
    ImageBase::swap(that);

    std::swap(_nCol, that._nCol);
    std::swap(_nRow, that._nRow);
    std::swap(_nSlc, that._nSlc);

    std::swap(_nColFT, that._nColFT);

    FOR_CELL_DIM_3
        std::swap(_box[k][j][i], that._box[k][j][i]);
}

Volume Volume::copyVolume() const
{
    Volume out;

    copyBase(out);

    out._nCol = _nCol;
    out._nRow = _nRow;
    out._nSlc = _nSlc;

    out._nColFT = _nColFT;

    FOR_CELL_DIM_3
        out._box[k][j][i] = _box[k][j][i];

    return out;
}

void Volume::alloc(int space)
{
    alloc(_nCol, _nRow, _nSlc, space);
}

void Volume::alloc(const int nCol,
                   const int nRow,
                   const int nSlc,
                   const int space)
{
    _nCol = nCol;
    _nRow = nRow;
    _nSlc = nSlc;

    if (space == RL_SPACE)
    {
        clearRL();

        _sizeRL = nCol * nRow * nSlc;
        _sizeFT = (nCol / 2 + 1) * nRow * nSlc;

#ifdef CXX11_PTR
        _dataRL.reset(new double[_sizeRL]);
#endif

#ifdef FFTW_PTR
#ifdef FFTW_PTR_THREAD_SAFETY
        #pragma omp critical
#endif
        _dataRL = (double*)fftw_malloc(_sizeRL * sizeof(double));

        if (_dataRL == NULL)
        {
            REPORT_ERROR("FAIL TO ALLOCATE SPACE");
            abort();
        }
#endif
    }
    else if (space == FT_SPACE)
    {
        clearFT();

        _sizeRL = nCol * nRow * nSlc;
        _sizeFT = (nCol / 2 + 1) * nRow * nSlc;

#ifdef CXX11_PTR
        _dataFT.reset(new Complex[_sizeFT]);
#endif

#ifdef FFTW_PTR
#ifdef FFTW_PTR_THREAD_SAFETY
        #pragma omp critical
#endif
        _dataFT = (Complex*)fftw_malloc(_sizeFT * sizeof(Complex));

        if (_dataFT == NULL)
        {
            REPORT_ERROR("FAIL TO ALLOCATE SPACE");
            abort();
        }
#endif
    }

    initBox();
}

double Volume::getRL(const int iCol,
                     const int iRow,
                     const int iSlc) const
{
    size_t index = iRL(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_RL(index);
#endif

    return _dataRL[index];
}

void Volume::setRL(const double value,
                   const int iCol,
                   const int iRow,
                   const int iSlc)
{
    size_t index = iRL(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_RL(index);
#endif

    _dataRL[index] = value;
}

void Volume::addRL(const double value,
                   const int iCol,
                   const int iRow,
                   const int iSlc)
{
    size_t index = iRL(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_RL(index);
#endif

    #pragma omp atomic
    _dataRL[index] += value;
}

Complex Volume::getFT(int iCol,
                      int iRow,
                      int iSlc) const
{
    bool conj;
    size_t index = iFT(conj, iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    return conj ? CONJUGATE(_dataFT[index]) : _dataFT[index];
}

Complex Volume::getFTHalf(const int iCol,
                          const int iRow,
                          const int iSlc) const
{
    size_t index = iFTHalf(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    return _dataFT[index];
}

void Volume::setFT(const Complex value,
                   int iCol,
                   int iRow,
                   int iSlc)
{
    bool conj;
    size_t index = iFT(conj, iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    _dataFT[index] = conj ? CONJUGATE(value) : value;
}

void Volume::setFTHalf(const Complex value,
                       const int iCol,
                       const int iRow,
                       const int iSlc)
{
    size_t index = iFTHalf(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    _dataFT[index] = value;
}

void Volume::addFT(const Complex value,
                   int iCol,
                   int iRow,
                   int iSlc)
{
    bool conj;
    size_t index = iFT(conj, iCol, iRow, iSlc);

    Complex val = conj ? CONJUGATE(value) : value;

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    #pragma omp atomic
    _dataFT[index].dat[0] += val.dat[0];
    #pragma omp atomic
    _dataFT[index].dat[1] += val.dat[1];
}

void Volume::addFTHalf(const Complex value,
                       const int iCol,
                       const int iRow,
                       const int iSlc)
{
    size_t index = iFTHalf(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    #pragma omp atomic
    _dataFT[index].dat[0] += value.dat[0];
    #pragma omp atomic
    _dataFT[index].dat[1] += value.dat[1];
}

void Volume::addFT(const double value,
                   int iCol,
                   int iRow,
                   int iSlc)
{
    size_t index = iFT(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    #pragma omp atomic
    _dataFT[index].dat[0] += value;
}

void Volume::addFTHalf(const double value,
                       const int iCol,
                       const int iRow,
                       const int iSlc)
{
    size_t index = iFTHalf(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    #pragma omp atomic
    _dataFT[index].dat[0] += value;
}

double Volume::getByInterpolationRL(const double iCol,
                                    const double iRow,
                                    const double iSlc,
                                    const int interp) const
{
    if (interp == NEAREST_INTERP)
        return getRL(AROUND(iCol), AROUND(iRow), AROUND(iSlc));

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};

    //WG_TRI_INTERP(w, x0, x, interp);
    WG_TRI_INTERP_LINEAR(w, x0, x);

    return getRL(w, x0);
}

Complex Volume::getByInterpolationFT(double iCol,
                                     double iRow,
                                     double iSlc,
                                     const int interp) const
{
    bool conj = conjHalf(iCol, iRow, iSlc);

    if (interp == NEAREST_INTERP)
    {
        Complex result = getFTHalf(AROUND(iCol), AROUND(iRow), AROUND(iSlc));

        return conj ? CONJUGATE(result) : result;
    }

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};

    //WG_TRI_INTERP(w, x0, x, interp);
    WG_TRI_INTERP_LINEAR(w, x0, x);

    Complex result = getFTHalf(w, x0);

    return conj ? CONJUGATE(result) : result;
}

void Volume::addFT(const Complex value,
                   double iCol,
                   double iRow,
                   double iSlc)
{
    bool conj = conjHalf(iCol, iRow, iSlc);

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};

    //WG_TRI_INTERP(w, x0, x, LINEAR_INTERP);
    WG_TRI_INTERP_LINEAR(w, x0, x);

    addFTHalf(conj ? CONJUGATE(value) : value,
              w,
              x0);
}

void Volume::addFT(const double value,
                   double iCol,
                   double iRow,
                   double iSlc)
{
    conjHalf(iCol, iRow, iSlc);

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};

    //WG_TRI_INTERP(w, x0, x, LINEAR_INTERP);
    WG_TRI_INTERP_LINEAR(w, x0, x);

    addFTHalf(value, w, x0);
}

void Volume::addFT(const Complex value,
                   const double iCol,
                   const double iRow,
                   const double iSlc,
                   const double a,
                   const double alpha)
{
    VOLUME_SUB_SPHERE_FT(a)
    {
        double r = NORM_3(iCol - i, iRow - j, iSlc - k);
        if (r < a) addFT(value * MKB_FT(r, a, alpha), i, j, k);
    }
}

void Volume::addFT(const double value,
                   const double iCol,
                   const double iRow,
                   const double iSlc,
                   const double a,
                   const double alpha)
{
    VOLUME_SUB_SPHERE_FT(a)
    {
        double r = NORM_3(iCol - i, iRow - j, iSlc - k);
        if (r < a) addFT(value * MKB_FT(r, a, alpha), i, j, k);
    }
}

void Volume::addFT(const Complex value,
                   const double iCol,
                   const double iRow,
                   const double iSlc,
                   const double a,
                   const TabFunction& kernel)
{
    double a2 = gsl_pow_2(a);

    VOLUME_SUB_SPHERE_FT(a)
    {
        double r2 = QUAD_3(iCol - i, iRow - j, iSlc - k);
        if (r2 < a2) addFT(value * kernel(r2), i, j, k);
    }
}

void Volume::addFT(const double value,
                   const double iCol,
                   const double iRow,
                   const double iSlc,
                   const double a,
                   const TabFunction& kernel)
{
    double a2 = gsl_pow_2(a);

    VOLUME_SUB_SPHERE_FT(a)
    {
        double r2 = QUAD_3(iCol - i, iRow - j, iSlc - k);
        if (r2 < a2) addFT(value * kernel(r2), i, j, k);
    }
}

void Volume::clear()
{
    ImageBase::clear();

    _nCol = 0;
    _nRow = 0;
    _nSlc = 0;
}

void Volume::initBox()
{
    _nColFT = _nCol / 2 + 1;

    FOR_CELL_DIM_3
        _box[k][j][i] = k * (_nColFT) * _nRow
                      + j * (_nColFT)
                      + i;
}

void Volume::coordinatesInBoundaryRL(const int iCol,
                                     const int iRow,
                                     const int iSlc) const
{
    if ((iCol < -_nCol / 2) || (iCol >= _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2) ||
        (iSlc < -_nSlc / 2) || (iSlc >= _nSlc / 2))
        REPORT_ERROR("ACCESSING VALUE OUT OF BOUNDARY");
}

void Volume::coordinatesInBoundaryFT(const int iCol,
                                     const int iRow,
                                     const int iSlc) const
{
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2) ||
        (iSlc < -_nSlc / 2) || (iSlc >= _nSlc / 2))
        REPORT_ERROR("ACCESSING VALUE OUT OF BOUNDARY");
}

double Volume::getRL(const double w[2][2][2],
                     const int x0[3]) const
{
    double result = 0;
    FOR_CELL_DIM_3 result += getRL(x0[0] + i, x0[1] + j, x0[2] + k)
                           * w[k][j][i];
    return result;
}

Complex Volume::getFTHalf(const double w[2][2][2],
                          const int x0[3]) const
{
    Complex result = COMPLEX(0, 0);

    if ((x0[1] != -1) &&
        (x0[2] != -1))
    {
        size_t index0 = iFTHalf(x0[0], x0[1], x0[2]);

        for (int i = 0; i < 8; i++)
        {
            size_t index = index0 + ((size_t*)_box)[i];

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
            BOUNDARY_CHECK_FT(index);
#endif

            result += _dataFT[index] * ((double*)w)[i];
        }
    }
    else
    {
        FOR_CELL_DIM_3 result += getFTHalf(x0[0] + i,
                                           x0[1] + j,
                                           x0[2] + k)
                               * w[k][j][i];
    }

    return result;
}

void Volume::addFTHalf(const Complex value,
                       const double w[2][2][2],
                       const int x0[3])
{
    if ((x0[1] != -1) &&
        (x0[2] != -1))
    {
        size_t index0 = iFTHalf(x0[0], x0[1], x0[2]);

        for (int i = 0; i < 8; i++)
        {
            size_t index = index0 + ((size_t*)_box)[i];

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
            BOUNDARY_CHECK_FT(index);
#endif
            
            #pragma omp atomic
            _dataFT[index].dat[0] += value.dat[0] * ((double*)w)[i];
            #pragma omp atomic
            _dataFT[index].dat[1] += value.dat[1] * ((double*)w)[i];
        }
    }
    else
    {
        FOR_CELL_DIM_3 addFTHalf(value * w[k][j][i],
                                 x0[0] + i,
                                 x0[1] + j,
                                 x0[2] + k);
    }
}

void Volume::addFTHalf(const double value,
                       const double w[2][2][2],
                       const int x0[3])
{
    if ((x0[1] != -1) &&
        (x0[2] != -1))
    {
        size_t index0 = iFTHalf(x0[0], x0[1], x0[2]);

        for (int i = 0; i < 8; i++)
        {
            size_t index = index0 + ((size_t*)_box)[i];

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
            BOUNDARY_CHECK_FT(index);
#endif

            #pragma omp atomic
            _dataFT[index].dat[0] += value * ((double*)w)[i];
        }
    }
    else
    {
        FOR_CELL_DIM_3 addFTHalf(value * w[k][j][i],
                                 x0[0] + i,
                                 x0[1] + j,
                                 x0[2] + k);
    }
}
