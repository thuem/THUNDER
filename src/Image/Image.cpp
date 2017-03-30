/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Image.h"

Image::Image() : _nCol(0), _nRow(0) {}

Image::Image(const int nCol,
             const int nRow,
             const int space)
{
    alloc(nCol, nRow, space);
}

Image::~Image() {}

void Image::swap(Image& that)
{
    ImageBase::swap(that);

    std::swap(_nCol, that._nCol);
    std::swap(_nRow, that._nRow);
}

Image Image::copyImage() const
{
    Image out;
    
    copyBase(out);

    out._nCol = _nCol;
    out._nRow = _nRow;

    return out;
}

void Image::alloc(const int space)
{
    alloc(_nCol, _nRow, space);
}

void Image::alloc(const int nCol,
                  const int nRow,
                  const int space)
{
    _nCol = nCol;
    _nRow = nRow;

    if (space == RL_SPACE)
    {
        clearRL();

        _sizeRL = nCol * nRow;
        _sizeFT = (nCol / 2 + 1) * nRow;

#ifdef CXX11_PTR
        _dataRL.reset(new double[_sizeRL]);
#endif

#ifdef FFTW_PTR
#ifdef FFTW_PTR_THREAD_SAFETY
        #pragma omp critical
#endif
        _dataRL = (double*)fftw_malloc(_sizeRL * sizeof(double));
#endif
    }
    else if (space == FT_SPACE)
    {
        clearFT();

        _sizeRL = nCol * nRow;
        _sizeFT = (nCol / 2 + 1) * nRow;

#ifdef CXX11_PTR
        _dataFT.reset(new Complex[_sizeFT]);
#endif

#ifdef FFTW_PTR
#ifdef FFTW_PTR_THREAD_SAFETY
        #pragma omp critical
#endif
        _dataFT = (Complex*)fftw_malloc(_sizeFT * sizeof(Complex));
#endif
    }
}

void Image::saveRLToBMP(const char* filename) const
{
    int nRowBMP = _nRow / 4 * 4;
    int nColBMP = _nCol / 4 * 4;

    float* image = new float[nRowBMP * nColBMP];

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Calculating Values in RL_BMP";
#endif

    for (int i = -nRowBMP / 2; i < nRowBMP / 2; i++)
        for (int j = -nColBMP / 2; j < nColBMP / 2; j++)
            image[(i + nRowBMP / 2)
                * nColBMP
                + (j + nColBMP / 2)] = _dataRL[(i >= 0 ? i : i + _nRow)
                                             * _nCol
                                             + (j >= 0 ? j : j + _nCol)];

    BMP bmp;

    if (bmp.open(filename, "wb") == 0)
        REPORT_ERROR("FAILING TO OPEN BITCAMP FILE");

    if (bmp.createBMP(image, nColBMP, nRowBMP) == false)
        REPORT_ERROR("FAILING TO CREATE BMP FILE");

    bmp.close();

    delete[] image;
}

void Image::saveFTToBMP(const char* filename, double c) const
{
    int nRowBMP = _nRow / 4 * 4;
    int nColBMP = _nCol / 4 * 4;

    float* image = new float[nRowBMP * nColBMP];

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Calculating Values in FT_BMP";
#endif

    for (int i = 0; i < nRowBMP; i++)
        for (int j = 0; j <= nColBMP / 2; j++)
        {
            double value = gsl_complex_abs2(_dataFT[(_nCol / 2 + 1) * i + j]);
            value = log(1 + value * c);

            int iImage = (i + nRowBMP / 2) % nRowBMP;
            int jImage = (j + nColBMP / 2) % nColBMP;
            image[nColBMP * iImage + jImage] = value;
        }

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Performing Hermite Symmetry";
#endif

    for (int i = 1; i < nRowBMP; i++)
        for (int j = 1; j < nColBMP / 2; j++)
        {
            size_t iDst = i * nColBMP + j;
            size_t iSrc = (nRowBMP - i + 1) * nColBMP - j;
            image[iDst] = image[iSrc];
        }

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Fixing Up the Missing Part";
#endif

    for (int j = 1; j < nColBMP / 2; j++)
    {
        int iDst = j;
        int iSrc = nColBMP - j;
        image[iDst] = image[iSrc];
    }

    BMP bmp;

    if (bmp.open(filename, "wb") == 0)
        REPORT_ERROR("FAILING TO OPEN BITCAMP FILE");
    if (bmp.createBMP(image, nColBMP, nRowBMP) == false)
        REPORT_ERROR("FAILING TO CREATE BMP FILE");

    bmp.close();

    delete[] image;
}

double Image::getRL(const int iCol,
                    const int iRow) const
{
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryRL(iCol, iRow);
#endif

    return _dataRL[iRL(iCol, iRow)];
}

void Image::setRL(const double value,
                  const int iCol,
                  const int iRow)
{
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryRL(iCol, iRow);
#endif

    _dataRL[iRL(iCol, iRow)] = value;
}

Complex Image::getFT(int iCol,
                     int iRow) const
{
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryFT(iCol, iRow);
#endif
    
    bool conj;
    int index = iFT(conj, iCol, iRow);

    return conj ? CONJUGATE(_dataFT[index]) : _dataFT[index];
}

Complex Image::getFTHalf(const int iCol,
                         const int iRow) const
{
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryFT(iCol, iRow);
#endif

    return _dataFT[iFTHalf(iCol, iRow)];
}

void Image::setFT(const Complex value,
                  int iCol,
                  int iRow)
{
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryFT(iCol, iRow);
#endif

    bool conj;
    int index = iFT(conj, iCol, iRow);

    _dataFT[index] = conj ? CONJUGATE(value) : value;
}

void Image::setFTHalf(const Complex value,
                      const int iCol,
                      const int iRow)
{
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryFT(iCol, iRow);
#endif

    _dataFT[iFTHalf(iCol, iRow)] = value;
}

void Image::addFT(const Complex value,
                  int iCol,
                  int iRow)
{
    bool conj;
    int index = iFT(conj, iCol, iRow);

    Complex val = conj ? CONJUGATE(value) : value;

    #pragma omp atomic
    _dataFT[index].dat[0] += val.dat[0];
    #pragma omp atomic
    _dataFT[index].dat[1] += val.dat[1];
}

void Image::addFTHalf(const Complex value,
                      const int iCol,
                      const int iRow)
{
    #pragma omp atomic
    _dataFT[iFTHalf(iCol, iRow)].dat[0] += value.dat[0];
    #pragma omp atomic
    _dataFT[iFTHalf(iCol, iRow)].dat[1] += value.dat[1];
}

void Image::addFT(const double value,
                  int iCol,
                  int iRow)
{
    #pragma omp atomic
    _dataFT[iFT(iCol, iRow)].dat[0] += value;
}

void Image::addFTHalf(const double value,
                      const int iCol,
                      const int iRow)
{
    #pragma omp atomic
    _dataFT[iFTHalf(iCol, iRow)].dat[0] += value;
}

double Image::getBiLinearRL(const double iCol,
                            const double iRow) const
{
    double w[2][2];
    int x0[2];
    double x[2] = {iCol, iRow};
    //WG_BI_INTERP(w, x0, x, LINEAR_INTERP);
    WG_BI_INTERP_LINEAR(w, x0, x);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryRL(x0[0], x0[1]);
    coordinatesInBoundaryRL(x0[0] + 1, x0[1] + 1);
#endif

    double result = 0;
    FOR_CELL_DIM_3 result += w[j][i] * getRL(x0[0] + i, x0[1] + j);
    return result;
}

Complex Image::getBiLinearFT(const double iCol,
                             const double iRow) const
{
    double w[2][2];
    int x0[2];
    double x[2] = {iCol, iRow};
    //WG_BI_INTERP(w, x0, x, LINEAR_INTERP);
    WG_BI_INTERP_LINEAR(w, x0, x);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryFT(x0[0], x0[1]);
    coordinatesInBoundaryFT(x0[0] + 1, x0[1] + 1);
#endif

    Complex result = COMPLEX(0, 0);
    FOR_CELL_DIM_2 result += w[j][i] * getFT(x0[0] + i , x0[1] + j);
    return result;
}

Complex Image::getByInterpolationFT(double iCol,
                                    double iRow,
                                    const int interp) const
{
    bool conj = conjHalf(iCol, iRow);

    if (interp == NEAREST_INTERP)
    {
        Complex result = getFTHalf(AROUND(iCol), AROUND(iRow));

        return conj ? CONJUGATE(result) : result;
    }

    double w[2][2];
    int x0[2];
    double x[2] = {iCol, iRow};

    //WG_BI_INTERP(w, x0, x, interp);
    WG_BI_INTERP_LINEAR(w, x0, x);

    Complex result = getFTHalf(w, x0);

    return conj ? CONJUGATE(result) : result;
}

void Image::addFT(const Complex value,
                  double iCol,
                  double iRow)
{
    bool conj = conjHalf(iCol, iRow);

    double w[2][2];
    int x0[2];
    double x[2] = {iCol, iRow};

    //WG_BI_INTERP(w, x0, x, LINEAR_INTERP);
    WG_BI_INTERP_LINEAR(w, x0, x);

    addFTHalf(conj ? CONJUGATE(value) : value,
              w,
              x0);
}

void Image::addFT(const double value,
                  double iCol,
                  double iRow)
{
    conjHalf(iCol, iRow);

    double w[2][2];
    int x0[2];
    double x[2] = {iCol, iRow};

    //WG_BI_INTERP(w, x0, x, LINEAR_INTERP);
    WG_BI_INTERP_LINEAR(w, x0, x);

    addFTHalf(value, w, x0);
}

void Image::coordinatesInBoundaryRL(const int iCol,
                                    const int iRow) const
{
    if ((iCol < -_nCol / 2) || (iCol >= _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2))
        REPORT_ERROR("ACCESSING VALUE OUT OF BOUNDARY");
}

void Image::coordinatesInBoundaryFT(const int iCol,
                                    const int iRow) const
{
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2))
        REPORT_ERROR("ACCESSING VALUE OUT OF BOUNDARY");
}

Complex Image::getFTHalf(const double w[2][2],
                         const int x0[2]) const
{
    Complex result = COMPLEX(0, 0);
    FOR_CELL_DIM_2 result += getFTHalf(x0[0] + i,
                                       x0[1] + j)
                           * w[j][i];
    return result;
}

void Image::addFTHalf(const Complex value,
                      const double w[2][2],
                      const int x0[2])
{
    FOR_CELL_DIM_2 addFTHalf(value * w[j][i],
                             x0[0] + i,
                             x0[1] + j);
                             
}

void Image::addFTHalf(const double value,
                      const double w[2][2],
                      const int x0[2])
{
    FOR_CELL_DIM_2 addFTHalf(value * w[j][i],
                             x0[0] + i,
                             x0[1] + j);
                             
}
