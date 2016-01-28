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
               const Space space)
{
    alloc(nCol, nRow, nSlc, space);
}

Volume::Volume(const Volume& that)
{
    *this = that;
}

Volume::~Volume()
{
    clear();
}

Volume& Volume::operator=(const Volume& that)
{
    ImageBase::operator=(that);

    _nCol = that.nColRL();
    _nRow = that.nRowRL();
    _nSlc = that.nSlcRL();

    return *this;
}

void Volume::alloc(Space space)
{
    alloc(_nCol, _nRow, _nSlc, space);
}

void Volume::alloc(const int nCol,
                   const int nRow,
                   const int nSlc,
                   const Space space)
{
    _nCol = nCol;
    _nRow = nRow;
    _nSlc = nSlc;

    if (space == realSpace)
    {
        clearRL();

        _sizeRL = nCol * nRow * nSlc;
        _sizeFT = (nCol / 2 + 1) * nRow * nSlc;

        _dataRL = new double[_sizeRL];
        if (_dataRL == NULL)
            REPORT_ERROR("Fail to allocate memory for storing volume");
    }
    else if (space == fourierSpace)
    {
        clearFT();

        _sizeRL = nCol * nRow * nSlc;
        _sizeFT = (nCol / 2 + 1) * nRow * nSlc;

        _dataFT = new Complex[_sizeFT];
        if (_dataFT == NULL)
            REPORT_ERROR("Fail to allocate memory for storing Fourier volume");
    }
}

int Volume::nColRL() const { return _nCol; }

int Volume::nRowRL() const { return _nRow; }

int Volume::nSlcRL() const { return _nSlc; }

int Volume::nColFT() const { return _nCol / 2 + 1; }

int Volume::nRowFT() const { return _nRow; }

int Volume::nSlcFT() const { return _nSlc; }

double Volume::getRL(const int iCol,
                     const int iRow,
                     const int iSlc) const
{
    coordinatesInBoundaryRL(iCol, iRow, iSlc);
    return _dataRL[VOLUME_INDEX(iCol, iRow, iSlc)];
}

void Volume::setRL(const double value,
                   const int iCol,
                   const int iRow,
                   const int iSlc)
{
    coordinatesInBoundaryRL(iCol, iRow, iSlc);
    _dataRL[VOLUME_INDEX(iCol, iRow, iSlc)] = value;
}

void Volume::addRL(const double value,
                   const int iCol,
                   const int iRow,
                   const int iSlc)
{
    coordinatesInBoundaryRL(iCol, iRow, iSlc);
    _dataRL[VOLUME_INDEX(iCol, iRow, iSlc)] += value;
}

Complex Volume::getFT(int iCol,
                      int iRow,
                      int iSlc,
                      const ConjugateFlag cf) const
{
    coordinatesInBoundaryFT(iCol, iRow, iSlc);
    bool flag;
    size_t index;
    VOLUME_FREQ_TO_STORE_INDEX(index, flag, iCol, iRow, iSlc, cf);
    return flag ? CONJUGATE(_dataFT[index]) : _dataFT[index];
}


void Volume::setFT(const Complex value,
                   int iCol,
                   int iRow,
                   int iSlc,
                   const ConjugateFlag cf)
{
    coordinatesInBoundaryFT(iCol, iRow, iSlc);
    bool flag;
    size_t index;
    VOLUME_FREQ_TO_STORE_INDEX(index, flag, iCol, iRow, iSlc, cf);
    _dataFT[index] = flag ? CONJUGATE(value) : value;
}

void Volume::addFT(const Complex value,
                   int iCol,
                   int iRow,
                   int iSlc,
                   const ConjugateFlag cf)
{
    coordinatesInBoundaryFT(iCol, iRow, iSlc);
    bool flag;
    size_t index;
    VOLUME_FREQ_TO_STORE_INDEX(index, flag, iCol, iRow, iSlc, cf);
    _dataFT[index] += flag ? CONJUGATE(value) : value;
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
    bool cf = VOLUME_CONJUGATE_HALF(iCol, iRow, iSlc);

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};
    switch (interp)
    {
        case NEAREST_INTERP: WG_TRI_NEAREST(w, x0, x); break;
        case LINEAR_INTERP: WG_TRI_LINEAR(w, x0, x); break;
        case SINC_INTERP: WG_TRI_SINC(w, x0, x); break;
    }

    Complex result = getFT(w, x0, conjugateNo);
    return cf ? CONJUGATE(result) : result;
}

void Volume::addFT(const Complex value,
                   const double iCol,
                   const double iRow,
                   const double iSlc,
                   const double a,
                   const double alpha)
{
    for (int k = MAX(-_nSlc / 2, floor(iSlc - a));
             k <= MIN(_nSlc / 2 - 1, ceil(iSlc + a));
             k++)
        for (int j = MAX(-_nRow / 2, floor(iRow - a));
                 j <= MIN(_nRow / 2 - 1, ceil(iRow + a));
                 j++)
            for (int i = MAX(-_nCol / 2, floor(iCol - a));
                     i <= MIN(_nCol / 2, ceil(iCol + a));
                     i++)
            {
                double r = NORM(iCol - i, iRow - j, iSlc - k);
                if (r < a) addFT(value * MKB_FT(r, a, alpha), i, j, k);
            }
}

void Volume::coordinatesInBoundaryRL(const int iCol,
                                     const int iRow,
                                     const int iSlc) const
{
    if ((iCol < 0) || (iCol >= _nCol) ||
        (iRow < 0) || (iRow >= _nRow) ||
        (iSlc < 0) || (iSlc >= _nSlc))
        REPORT_ERROR("Try to get value out of the boundary");
}

void Volume::coordinatesInBoundaryFT(const int iCol,
                                     const int iRow,
                                     const int iSlc) const
{
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2) ||
        (iSlc < -_nSlc / 2) || (iSlc >= _nSlc / 2))
        REPORT_ERROR("Try to get value out of the boundary");
}

double Volume::getRL(const double w[2][2][2],
                     const int x0[3]) const
{
    double result = 0;
    FOR_CELL_DIM_3 result += getRL(x0[0] + i, x0[1] + j, x0[2] + k)
                           * w[i][j][k];
    return result;
}

Complex Volume::getFT(const double w[2][2][2],
                      const int x0[3],
                      const ConjugateFlag conjugateFlag) const
{
    Complex result = COMPLEX(0, 0);
    FOR_CELL_DIM_3 result += getFT(x0[0] + i,
                                   x0[1] + j,
                                   x0[2] + k,
                                   conjugateFlag)
                           * w[i][j][k];
    return result;
}
