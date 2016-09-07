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

Volume::Volume() {}

Volume::Volume(const int nCol,
               const int nRow,
               const int nSlc,
               const int space)
{
    alloc(nCol, nRow, nSlc, space);
}

Volume::~Volume() {}

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

        _dataRL.reset(new double[_sizeRL]);
    }
    else if (space == FT_SPACE)
    {
        clearFT();

        _sizeRL = nCol * nRow * nSlc;
        _sizeFT = (nCol / 2 + 1) * nRow * nSlc;

        _dataFT.reset(new Complex[_sizeFT]);
    }
}

double Volume::getRL(const int iCol,
                     const int iRow,
                     const int iSlc) const
{
    // coordinatesInBoundaryRL(iCol, iRow, iSlc);
    
    return _dataRL[iRL(iCol, iRow, iSlc)];
}

void Volume::setRL(const double value,
                   const int iCol,
                   const int iRow,
                   const int iSlc)
{
    // coordinatesInBoundaryRL(iCol, iRow, iSlc);

    _dataRL[iRL(iCol, iRow, iSlc)] = value;
}

void Volume::addRL(const double value,
                   const int iCol,
                   const int iRow,
                   const int iSlc)
{
    // coordinatesInBoundaryRL(iCol, iRow, iSlc);

    #pragma omp atomic
    _dataRL[iRL(iCol, iRow, iSlc)] += value;
}

Complex Volume::getFT(int iCol,
                      int iRow,
                      int iSlc) const
{
    // coordinatesInBoundaryFT(iCol, iRow, iSlc);

    bool conj;
    int index = iFT(conj, iCol, iRow, iSlc);

    return conj ? CONJUGATE(_dataFT[index]) : _dataFT[index];
}

Complex Volume::getFTHalf(const int iCol,
                          const int iRow,
                          const int iSlc) const
{
    return _dataFT[iFTHalf(iCol, iRow, iSlc)];
}

void Volume::setFT(const Complex value,
                   int iCol,
                   int iRow,
                   int iSlc)
{
    // coordinatesInBoundaryFT(iCol, iRow, iSlc);

    bool conj;
    int index = iFT(conj, iCol, iRow, iSlc);

    _dataFT[index] = conj ? CONJUGATE(value) : value;
}

void Volume::setFTHalf(const Complex value,
                       const int iCol,
                       const int iRow,
                       const int iSlc)
{
    _dataFT[iFTHalf(iCol, iRow, iSlc)] = value;
}

void Volume::addFT(const Complex value,
                   int iCol,
                   int iRow,
                   int iSlc)
{
    bool conj;
    int index = iFT(conj, iCol, iRow, iSlc);

    Complex val = conj ? CONJUGATE(value) : value;

    #pragma omp atomic
    _dataFT[index].dat[0] += val.dat[0];
    #pragma omp atomic
    _dataFT[index].dat[1] += val.dat[1];
}

void Volume::addFT(const double value,
                   int iCol,
                   int iRow,
                   int iSlc)
{
    int index = iFT(iCol, iRow, iSlc);

    #pragma omp atomic
    _dataFT[index].dat[0] += value;
}

double Volume::getByInterpolationRL(const double iCol,
                                    const double iRow,
                                    const double iSlc,
                                    const int interp) const
{
    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};
    switch (interp)
    {
        case NEAREST_INTERP: WG_TRI_NEAREST(w, x0, x); break;
        case LINEAR_INTERP: WG_TRI_LINEAR(w, x0, x); break;
        case SINC_INTERP: WG_TRI_SINC(w, x0, x); break;
    }

    return getRL(w, x0);
}

Complex Volume::getByInterpolationFT(double iCol,
                                     double iRow,
                                     double iSlc,
                                     const int interp) const
{
    bool conj = conjHalf(iCol, iRow, iSlc);

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};

    switch (interp)
    {
        case NEAREST_INTERP: WG_TRI_NEAREST(w, x0, x); break;
        case LINEAR_INTERP: WG_TRI_LINEAR(w, x0, x); break;
        case SINC_INTERP: WG_TRI_SINC(w, x0, x); break;
    }

    Complex result = getFTHalf(w, x0);

    return conj ? CONJUGATE(result) : result;
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

Volume Volume::copyVolume() const
{
    Volume out;

    copyBase(out);
    out._nCol = _nCol;
    out._nRow = _nRow;
    out._nSlc = _nSlc;

    return out;
}

void Volume::coordinatesInBoundaryRL(const int iCol,
                                     const int iRow,
                                     const int iSlc) const
{
    if ((iCol < -_nCol / 2) || (iCol >= _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2) ||
        (iSlc < -_nSlc / 2) || (iSlc >= _nSlc / 2))
        CLOG(FATAL, "LOGGER_SYS") << "Accessing Value out of Boundary";
}

void Volume::coordinatesInBoundaryFT(const int iCol,
                                     const int iRow,
                                     const int iSlc) const
{
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2) ||
        (iSlc < -_nSlc / 2) || (iSlc >= _nSlc / 2))
        CLOG(FATAL, "LOGGER_SYS") << "Accessing Value out of Boundary";
}

double Volume::getRL(const double w[2][2][2],
                     const int x0[3]) const
{
    double result = 0;
    FOR_CELL_DIM_3 result += getRL(x0[0] + i, x0[1] + j, x0[2] + k)
                           * w[i][j][k];
    return result;
}

Complex Volume::getFTHalf(const double w[2][2][2],
                          const int x0[3]) const
{
    Complex result = COMPLEX(0, 0);
    FOR_CELL_DIM_3 result += getFTHalf(x0[0] + i,
                                       x0[1] + j,
                                       x0[2] + k)
                           * w[i][j][k];
    return result;
}
