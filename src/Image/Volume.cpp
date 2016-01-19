/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency: VolumeBase.h
 * Test:
 * Execution:
 * Description: a volume class
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
    _dataRL[VOLUME_INDEX(iCol, iRow, iSlc)];
}

Complex Volume::getFT(int iCol,
                      int iRow,
                      int iSlc,
                      const ConjugateFlag cf) const
{
    coordinatesInBoundaryFT(iCol, iRow, iSlc);
    bool flag;
    size_t index = VOLUME_FREQ_TO_STORE_INDEX(iCol, iRow, iSlc, cf);
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
    size_t index = VOLUME_FREQ_TO_STORE_INDEX(iCol, iRow, iSlc, cf);
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
    size_t index = VOLUME_FREQ_TO_STORE_INDEX(iCol, iRow, iSlc, cf);
    _dataFT[index] += flag ? CONJUGATE(value) : value;
}

double Volume::getByInterpolationRL(const double iCol,
                                    const double iRow, 
                                    const double iSlc,
                                    const Interpolation3DStyle style) const
{
    // calculate grids and weights
    int x0, y0, z0;
    double w000, w001, w010, w011, w100, w101, w110, w111; 
    interpolationWeight3DGrid(w000, w001, w010, w011,
                              w100, w101, w110, w111,
                              x0, y0, z0,
                              iCol, iRow, iSlc, style);

    // get value by weights
    // get method inclues boundary check.
    return getRL(x0, y0, z0,
                 w000, w001, w010, w011,
                 w100, w101, w110, w111);
}

Complex Volume::getByInterpolationFT(double iCol,
                                     double iRow,
                                     double iSlc,
                                     const Interpolation3DStyle style) const
{
    bool cf = VOLUME_CONJUGATE_HALF(iCol, iRow, iSlc);
    // ConjugateFlag conjugateFlag = conjugate(iCol, iRow, iSlc);

    // calculate grids and weights
    int x0, y0, z0;
    double w000, w001, w010, w011, w100, w101, w110, w111; 
    interpolationWeight3DGrid(w000, w001, w010, w011,
                              w100, w101, w110, w111,
                              x0, y0, z0,
                              iCol, iRow, iSlc, style);

    /***
    printf("x0 = %d\n", x0);
    printf("y0 = %d\n", y0);
    printf("z0 = %d\n", z0);
    printf("w000 = %f\n", w000);
    printf("w001 = %f\n", w001);
    printf("w010 = %f\n", w010);
    printf("w011 = %f\n", w011);
    printf("w100 = %f\n", w100);
    printf("w101 = %f\n", w101);
    printf("w110 = %f\n", w110);
    printf("w111 = %f\n", w111);
    ***/
    // get value from Fourier space by weights
    // getFT method inclues boundary check.
    Complex result = getFT(x0, y0, z0,
                           w000, w001, w010, w011,
                           w100, w101, w110, w111, conjugateNo);

    return cf ? CONJUGATE(result) : result;
}

void Volume::addByInterpolationFT(const Complex value,
                                  double iCol,
                                  double iRow,
                                  double iSlc,
                                  double* weight,
                                  const Interpolation3DStyle style)
{
    bool cf = VOLUME_CONJUGATE_HALF(iCol, iRow, iSlc);

    // calculate grids and weights
    int x0, y0, z0;
    double w000, w001, w010, w011, w100, w101, w110, w111; 
    interpolationWeight3DGrid(w000, w001, w010, w011,
                              w100, w101, w110, w111,
                              x0, y0, z0,
                              iCol, iRow, iSlc, style);
  
    // add into the volume
    // addFT method includes boundary check.
    addFT(cf ? CONJUGATE(value) : value,
          x0, y0, z0,
          w000, w001, w010, w011,
          w100, w101, w110, w111, conjugateNo, weight);
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
    /***
    printf("x0 = %d\n", iCol);
    printf("y0 = %d\n", iRow);
    printf("z0 = %d\n", iSlc);
    ***/
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2) ||
        (iSlc < -_nSlc / 2) || (iSlc >= _nSlc / 2))
        REPORT_ERROR("Try to get value out of the boundary");
}

double Volume::getRL(const int x0, const int y0, const int z0,
                     const double w000, const double w001,
                     const double w010, const double w011,
                     const double w100, const double w101,
                     const double w110, const double w111) const
{
    return getRL(x0, y0, z0) * w000
         + getRL(x0, y0, z0 + 1) * w001
         + getRL(x0, y0 + 1, z0) * w010
         + getRL(x0, y0 + 1, z0 + 1) * w011
         + getRL(x0 + 1, y0, z0) * w100
         + getRL(x0 + 1, y0, z0 + 1) * w101
         + getRL(x0 + 1, y0 + 1, z0) * w110
         + getRL(x0 + 1, y0 + 1, z0 + 1) * w111;
}

Complex Volume::getFT(const int x0, const int y0, const int z0,
                      const double w000, const double w001,
                      const double w010, const double w011,
                      const double w100, const double w101,
                      const double w110, const double w111,
                      const ConjugateFlag conjugateFlag) const
{
    /***
    printf("x0 = %d\n", x0);
    printf("y0 = %d\n", y0);
    printf("z0 = %d\n", z0);
    ***/
    Complex result = getFT(x0, y0, z0, conjugateFlag) * w000;
    result += getFT(x0, y0, z0 + 1, conjugateFlag) * w001;
    result += getFT(x0, y0 + 1, z0, conjugateFlag) * w010; 
    result += getFT(x0, y0 + 1, z0 + 1, conjugateFlag) * w011; 
    result += getFT(x0 + 1, y0, z0, conjugateFlag) * w100;
    result += getFT(x0 + 1, y0, z0 + 1, conjugateFlag) * w101; 
    result += getFT(x0 + 1, y0 + 1, z0, conjugateFlag) * w110; 
    result += getFT(x0 + 1, y0 + 1, z0 + 1, conjugateFlag) * w111; 

    return result;
}

void Volume::addFT(const Complex value,
                   const int x0, const int y0, const int z0,
                   const double w000, const double w001,
                   const double w010, const double w011,
                   const double w100, const double w101,
                   const double w110, const double w111,
                   const ConjugateFlag conjugateFlag,
                   double* weight)
{
    addFT(value * w000, x0, y0, z0, conjugateFlag);
    addFT(value * w001, x0, y0, z0 + 1, conjugateFlag);
    addFT(value * w010, x0, y0 + 1, z0, conjugateFlag);
    addFT(value * w011, x0, y0 + 1, z0 + 1, conjugateFlag);
    addFT(value * w100, x0 + 1, y0, z0, conjugateFlag);
    addFT(value * w101, x0 + 1, y0, z0 + 1, conjugateFlag);
    addFT(value * w110, x0 + 1, y0 + 1, z0, conjugateFlag);
    addFT(value * w111, x0 + 1, y0 + 1, z0 + 1, conjugateFlag);

    if (weight != NULL)
    {
        weight[(VOLUME_FREQ_TO_STORE_INDEX_HALF(x0, y0, z0))] += w000;
        weight[(VOLUME_FREQ_TO_STORE_INDEX_HALF(x0, y0, z0 + 1))] += w001;
        weight[(VOLUME_FREQ_TO_STORE_INDEX_HALF(x0, y0 + 1, z0))] += w010;
        weight[(VOLUME_FREQ_TO_STORE_INDEX_HALF(x0, y0 + 1, z0 + 1))] += w011;
        weight[(VOLUME_FREQ_TO_STORE_INDEX_HALF(x0 + 1, y0, z0))] += w100;
        weight[(VOLUME_FREQ_TO_STORE_INDEX_HALF(x0 + 1, y0, z0 + 1))] += w101;
        weight[(VOLUME_FREQ_TO_STORE_INDEX_HALF(x0 + 1, y0 + 1, z0))] += w110;
        weight[(VOLUME_FREQ_TO_STORE_INDEX_HALF(x0 + 1, y0 + 1, z0 + 1))] += w111;
        /***
        weight[getIndexFT(x0, y0, z0)] += w000;
        weight[getIndexFT(x0, y0, z0 + 1)] += w001; 
        weight[getIndexFT(x0, y0 + 1, z0)] += w010; 
        weight[getIndexFT(x0, y0 + 1, z0 + 1)] += w011; 
        weight[getIndexFT(x0 + 1, y0, z0)] += w100;
        weight[getIndexFT(x0 + 1, y0, z0 + 1)] += w101; 
        weight[getIndexFT(x0 + 1, y0 + 1, z0)] += w110; 
        weight[getIndexFT(x0 + 1, y0 + 1, z0 + 1)] += w111; 
        ***/
    }
}
