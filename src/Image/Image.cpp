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
             const Space space)
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

void Image::alloc(const Space space)
{
    alloc(_nCol, _nRow, space);
}

void Image::alloc(const int nCol,
                  const int nRow,
                  const Space space)
{
    _nCol = nCol;
    _nRow = nRow;

    _sizeRL = nCol * nRow;
    _sizeFT = (nCol / 2 + 1) * nRow;

    if (space == realSpace)
    {
        clearRL();
        _dataRL = new double[_sizeRL];
        if (_dataRL == NULL)
            REPORT_ERROR("Fail to allocate memory for storing image");
    }
    else if (space == fourierSpace)
    {
        clearFT();
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

    for (size_t i = 0; i < _sizeRL; i++)
        image[i] = (float)_dataRL[i];

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
    return _dataRL[IMAGE_INDEX(iCol, iRow)];
}

void Image::setRL(const double value,
                  const int iCol,
                  const int iRow)
{
    coordinatesInBoundaryRL(iCol, iRow);
    _dataRL[IMAGE_INDEX(iCol, iRow)] = value;
}

Complex Image::getFT(int iCol,
                     int iRow) const
{
    coordinatesInBoundaryFT(iCol, iRow);
    size_t index;
    bool cf = IMAGE_FREQ_TO_STORE_INDEX(iCol, iRow);
    return cf ? CONJUGATE(_dataFT[index]) : _dataFT[index];
}

void Image::setFT(const Complex value,
                  int iCol,
                  int iRow)
{
    coordinatesInBoundaryFT(iCol, iRow);
    size_t index;
    bool cf = IMAGE_FREQ_TO_STORE_INDEX(iCol, iRow);
    _dataFT[index] = cf ? CONJUGATE(value) : value;
}

double Image::getBiLinearRL(const double iCol,
                            const double iRow) const
{
    double w00, w01, w10, w11;
    int x0, y0;
    biLinearWeightGrid(w00, w01, w10, w11, x0, y0, iCol, iRow);

    coordinatesInBoundaryRL(x0, y0);
    coordinatesInBoundaryRL(x0 + 1, y0 + 1);

    return w00 * getRL(x0, y0)
         + w01 * getRL(x0, y0 + 1)
         + w10 * getRL(x0 + 1, y0)
         + w11 * getRL(x0 + 1, y0 + 1);
}

Complex Image::getBiLinearFT(const double iCol,
                             const double iRow) const
{
    double w00, w01, w10, w11;
    int x0, y0;
    biLinearWeightGrid(w00, w01, w10, w11, x0, y0, iCol, iRow);

    coordinatesInBoundaryFT(x0, y0);
    coordinatesInBoundaryFT(x0 + 1, y0 + 1);

    return w00 * getFT(x0, y0)
         + w01 * getFT(x0, y0 + 1)
         + w10 * getFT(x0 + 1, y0)
         + w11 * getFT(x0 + 1, y0 + 1);
}

void Image::coordinatesInBoundaryRL(const int iCol,
                                    const int iRow) const
{
    if ((iCol < 0) || (iCol >= _nCol) ||
        (iRow < 0) || (iRow >= _nRow))
        REPORT_ERROR("Try to get value out of the boundary");
}

void Image::coordinatesInBoundaryFT(const int iCol,
                                    const int iRow) const
{
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2))
        REPORT_ERROR("Try to get FT value out of the boundary");
}
