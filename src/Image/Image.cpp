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

Image::~Image() {}

Image::Image(const int nCol,
             const int nRow,
             const int space)
{
    alloc(nCol, nRow, space);
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
        //_dataRL = fftw_alloc_real(_sizeRL);
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
        //_dataFT = (Complex*)fftw_alloc_complex(_sizeFT);
        _dataFT = (Complex*)fftw_malloc(_sizeFT * sizeof(Complex));
#endif
    }
}

void Image::saveRLToBMP(const char* filename) const
{
    int nRowBMP = _nRow / 4 * 4;
    int nColBMP = _nCol / 4 * 4;

    //float* image = new float[_sizeRL];

    float* image = new float[nRowBMP * nColBMP];

    /***
    for (int i = 0; i < _nRow; i++)
        for (int j = 0; j < _nCol; j++)
            image[(i + _nRow / 2) % _nRow * _nCol
                 +(j + _nCol / 2) % _nCol] = _dataRL[i * _nCol + j];
                 ***/

    for (int i = -nRowBMP / 2; i < nRowBMP / 2; i++)
        for (int j = -nColBMP / 2; j < nColBMP / 2; j++)
            image[(i + nRowBMP / 2)
                * nColBMP
                + (j + nColBMP / 2)] = _dataRL[(i >= 0 ? i : i + _nRow)
                                             * _nCol
                                             + (j >= 0 ? j : j + _nCol)];

    BMP bmp;

    if (bmp.open(filename, "wb") == 0)
        REPORT_ERROR("Fail to open bitcamp file.");

    if (bmp.createBMP(image, nColBMP, nRowBMP) == false)
        REPORT_ERROR("Fail to create BMP image.");

    bmp.close();

    delete[] image;
}

void Image::saveFTToBMP(const char* filename, double c) const
{
    int nRowBMP = _nRow / 4 * 4;
    int nColBMP = _nCol / 4 * 4;

    // CLOG(INFO, "LOGGER_SYS") << "_sizeRL = " << _sizeRL;

    //float* image = new float[_sizeRL];
    float* image = new float[nRowBMP * nColBMP];

    // CLOG(INFO, "LOGGER_SYS") << "Calculating Values in FT_BMP";

    /***
    for (int i = 0; i < _nRow; i++)
        for (int j = 0; j <= _nCol / 2; j++)
        {
            double value = gsl_complex_abs2(_dataFT[(_nCol / 2 + 1) * i + j]);
            value = log(1 + value * c);

            int iImage = (i + _nRow / 2 ) % _nRow;
            int jImage = (j + _nCol / 2 ) % _nCol;
            image[_nCol * iImage + jImage] = value;
        }
        ***/

    for (int i = 0; i < nRowBMP; i++)
        for (int j = 0; j <= nColBMP / 2; j++)
        {
            double value = gsl_complex_abs2(_dataFT[(_nCol / 2 + 1) * i + j]);
            value = log(1 + value * c);

            int iImage = (i + nRowBMP / 2) % nRowBMP;
            int jImage = (j + nColBMP / 2) % nColBMP;
            image[nColBMP * iImage + jImage] = value;
        }

    // CLOG(INFO, "LOGGER_SYS") << "Performing Hermite Symmetry";

    for (int i = 1; i < nRowBMP; i++)
        for (int j = 1; j < nColBMP / 2; j++)
        {
            size_t iDst = i * nColBMP + j;
            size_t iSrc = (nRowBMP - i + 1) * nColBMP - j;
            image[iDst] = image[iSrc];
        }

    // CLOG(INFO, "LOGGER_SYS") << "Fixing Up the Missing Part";

    for (int j = 1; j < nColBMP / 2; j++)
    {
        int iDst = j;
        int iSrc = nColBMP - j;
        image[iDst] = image[iSrc];
    }

    BMP bmp;

    if (bmp.open(filename, "wb") == 0)
        REPORT_ERROR("Fail to open bitcamp file.");
    if (bmp.createBMP(image, nColBMP, nRowBMP) == false)
        REPORT_ERROR("Fail to create BMP image.");
    bmp.close();

    delete[] image;
}

double Image::getRL(const int iCol,
                    const int iRow) const
{
    // coordinatesInBoundaryRL(iCol, iRow);

    return _dataRL[iRL(iCol, iRow)];
}

void Image::setRL(const double value,
                  const int iCol,
                  const int iRow)
{
    // coordinatesInBoundaryRL(iCol, iRow);

    _dataRL[iRL(iCol, iRow)] = value;
}

Complex Image::getFT(int iCol,
                     int iRow) const
{
    // coordinatesInBoundaryFT(iCol, iRow);
    
    bool conj;
    int index = iFT(conj, iCol, iRow);

    return conj ? CONJUGATE(_dataFT[index]) : _dataFT[index];
}

Complex Image::getFTHalf(const int iCol,
                         const int iRow) const
{
    return _dataFT[iFTHalf(iCol, iRow)];
}

void Image::setFT(const Complex value,
                  int iCol,
                  int iRow)
{
    // coordinatesInBoundaryFT(iCol, iRow);

    bool conj;
    int index = iFT(conj, iCol, iRow);

    _dataFT[index] = conj ? CONJUGATE(value) : value;
}

double Image::getBiLinearRL(const double iCol,
                            const double iRow) const
{
    double w[2][2];
    int x0[2];
    double x[2] = {iCol, iRow};
    WG_BI_INTERP(w, x0, x, LINEAR_INTERP);
    // coordinatesInBoundaryRL(x0[0], x0[1]);
    // coordinatesInBoundaryRL(x0[0] + 1, x0[1] + 1);

    double result = 0;
    FOR_CELL_DIM_3 result += w[i][j] * getRL(x0[0] + i, x0[1] + j);
    return result;
}

Complex Image::getBiLinearFT(const double iCol,
                             const double iRow) const
{
    double w[2][2];
    int x0[2];
    double x[2] = {iCol, iRow};
    WG_BI_INTERP(w, x0, x, LINEAR_INTERP);

    // coordinatesInBoundaryFT(x0[0], x0[1]);
    // coordinatesInBoundaryFT(x0[0] + 1, x0[1] + 1);

    Complex result = COMPLEX(0, 0);
    FOR_CELL_DIM_2 result += w[i][j] * getFT(x0[0] + i , x0[1] + j);
    return result;
}

void Image::coordinatesInBoundaryRL(const int iCol,
                                    const int iRow) const
{
    if ((iCol < -_nCol / 2) || (iCol >= _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2))
        CLOG(FATAL, "LOGGER_SYS") << "Accessing Value out of Boundary";
}

void Image::coordinatesInBoundaryFT(const int iCol,
                                    const int iRow) const
{
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2))
        CLOG(FATAL, "LOGGER_SYS") << "Accessing Value out of Boundary";
}
