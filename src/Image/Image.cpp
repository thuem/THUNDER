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

Image::Image() {}

Image::Image(const int nCol,
             const int nRow,
             const int space)
{
    alloc(nCol, nRow, space);
}

Image::Image(const Image& that)
{
    *this = that;
}

Image::~Image()
{
    clear();
}

Image& Image::operator=(const Image& that)
{
    ImageBase::operator=(that);

    _nCol = that.nColRL();
    _nRow = that.nRowRL();

    return *this;
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
        _dataRL = new double[_sizeRL];
        if (_dataRL == NULL)
            REPORT_ERROR("Fail to allocate memory for storing image");
    }
    else if (space == FT_SPACE)
    {
        clearFT();
        _sizeRL = nCol * nRow;
        _sizeFT = (nCol / 2 + 1) * nRow;
        _dataFT = new Complex[_sizeFT];
        if (_dataFT == NULL)
            REPORT_ERROR("Fail to allocate memory for storing Fourier image");
    }
}

int Image::nColRL() const { return _nCol; }

int Image::nRowRL() const { return _nRow; }

int Image::nColFT() const { return _nCol / 2 + 1; }

int Image::nRowFT() const { return _nRow; }

void Image::saveRLToBMP(const char* filename) const
{
    float* image = new float[_sizeRL];

    for (int i = 0; i < _nRow; i++)
        for (int j = 0; j < _nCol; j++)
            image[(i + _nRow / 2) % _nRow * _nCol
                 +(j + _nCol / 2) % _nCol] = _dataRL[i * _nCol + j];

    BMP bmp;

    if (bmp.open(filename, "wb") == 0)
        REPORT_ERROR("Fail to open bitcamp file.");

    if (bmp.createBMP(image, _nCol, _nRow) == false)
        REPORT_ERROR("Fail to create BMP image.");

    bmp.close();

    delete[] image;
}

void Image::saveFTToBMP(const char* filename, double c) const
{
    float* image = new float[_sizeRL];

    for (int i = 0; i < _nRow; i++)
        for (int j = 0; j <= _nCol / 2; j++)
        {
            double value = gsl_complex_abs2(_dataFT[(_nCol / 2 + 1) * i + j]);
            value = log(1 + value * c);
                
            int iImage = (i + _nRow / 2 ) % _nRow;
            int jImage = (j + _nCol / 2 ) % _nCol;
            image[_nCol * iImage + jImage] = value;
        }   

    for (int i = 1; i < _nRow; i++)
        for (int j = 1; j < _nCol / 2; j++)
        {
            size_t iDst = i * _nCol + j;
            size_t iSrc = (_nRow - i + 1) * _nCol - j;
            image[iDst] = image[iSrc];
        }

    for (int j = 1; j < _nCol / 2; j++)
    {
        int iDst = j;
        int iSrc = _nCol - j;
        image[iDst] = image[iSrc];
    }

    BMP bmp;

    if (bmp.open(filename, "wb") == 0)
        REPORT_ERROR("Fail to open bitcamp file.");
    if (bmp.createBMP(image, _nCol, _nRow) == false)
        REPORT_ERROR("Fail to create BMP image.");
    bmp.close();

    delete[] image;
}

double Image::getRL(const int iCol,
                    const int iRow) const
{
    coordinatesInBoundaryRL(iCol, iRow);
    return _dataRL[IMAGE_INDEX_RL((iCol >= 0) ? iCol : iCol + _nCol,
                                  (iRow >= 0) ? iRow : iRow + _nRow)];
}

void Image::setRL(const double value,
                  const int iCol,
                  const int iRow)
{
    coordinatesInBoundaryRL(iCol, iRow);
    _dataRL[IMAGE_INDEX_RL((iCol >= 0) ? iCol : iCol + _nCol,
                           (iRow >= 0) ? iRow : iRow + _nRow)] = value;
}

Complex Image::getFT(int iCol,
                     int iRow) const
{
    coordinatesInBoundaryFT(iCol, iRow);
    size_t index;
    bool cf;
    IMAGE_FREQ_TO_STORE_INDEX(index, cf, iCol, iRow);
    return cf ? CONJUGATE(_dataFT[index]) : _dataFT[index];
}

void Image::setFT(const Complex value,
                  int iCol,
                  int iRow)
{
    coordinatesInBoundaryFT(iCol, iRow);
    size_t index;
    bool cf;
    IMAGE_FREQ_TO_STORE_INDEX(index, cf, iCol, iRow);
    _dataFT[index] = cf ? CONJUGATE(value) : value;
}

double Image::getBiLinearRL(const double iCol,
                            const double iRow) const
{
    double w[2][2];
    int x0[2];
    double x[2] = {iCol, iRow};
    WG_BI_LINEAR(w, x0, x);

    coordinatesInBoundaryRL(x0[0], x0[1]);
    coordinatesInBoundaryRL(x0[0] + 1, x0[1] + 1);

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
    WG_BI_LINEAR(w, x0, x);

    coordinatesInBoundaryFT(x0[0], x0[1]);
    coordinatesInBoundaryFT(x0[0] + 1, x0[1] + 1);

    Complex result = COMPLEX(0, 0);
    FOR_CELL_DIM_2 result += w[i][j] * getFT(x0[0] + i , x0[1] + j);
    return result;
}

void Image::coordinatesInBoundaryRL(const int iCol,
                                    const int iRow) const
{
    if ((iCol < -_nCol / 2) || (iCol >= _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2))
        REPORT_ERROR("Try to get value out of the boundary");
}

void Image::coordinatesInBoundaryFT(const int iCol,
                                    const int iRow) const
{
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2))
        REPORT_ERROR("Try to get FT value out of the boundary");
}
