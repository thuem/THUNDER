/***********************************************************************
 * FileName: Image.cu
 * Author  : Kunpeng WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#include "Image.cuh"

namespace cuthunder {

///////////////////////////////////////////////////////////////////////

HD_CALLABLE void Image::init(Complex* data, int nCol, int nRow)
{
    _nCol = nCol;
    _nRow = nRow;

    _size = (nCol / 2 + 1) * nRow;

    /* Load from Host */
    _dataFT = data;
}

HD_CALLABLE int Image::getIndex(int i, int j, bool& flag) const
{
    if (i >= 0) flag = false;
    else { i *= -1; j *= -1; flag = true; }

    if (j < 0) j += _nRow;

    return CU_IMAGE_INDEX_FT(i, j);
}

HD_CALLABLE Complex Image::getFT(const int iCol,
                                 const int iRow) const
{
    if(!coordinatesInBoundaryFT(iCol, iRow))
        return Complex();
    
    bool flag;
    int index = getIndex(iCol, iRow, flag);

    if (flag)
        return _dataFT[index].conj();
    else
        return _dataFT[index];
}

HD_CALLABLE Complex* Image::getFTData() const
{
    return _dataFT;
}

HD_CALLABLE void Image::setFT(Complex value,
                              const int iCol,
                              const int iRow)
{
    if(!coordinatesInBoundaryFT(iCol, iRow))
        return ;
    
    bool flag;
    int index = getIndex(iCol, iRow, flag);
    flag ? value.conj() : value;

    _dataFT[index] = value;
}

HD_CALLABLE void Image::addFT(Complex value,
                              const int iCol,
                              const int iRow)
{
    if(!coordinatesInBoundaryFT(iCol, iRow))
        return ;
    
    bool flag;
    int index = getIndex(iCol, iRow, flag);
    flag ? value.conj() : value;

    _dataFT[index] += value;
}

HD_CALLABLE bool Image::coordinatesInBoundaryFT(const int iCol,
                                                const int iRow) const
{
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2))
    {
        printf("***Image out of boundary!\n");
        return false;
    }
    else
        return true;
}

HD_CALLABLE void displayImageFT(Image& img)
{
    double* data = (double*)img.getFTData();

    if (data == NULL)
    {
        printf("***It's an empty image!\n");
        return ;
    }

    int dimX = img.nColFT();
    int dimY = img.nRowFT();

    for (int j = 0; j < dimY; ++j)
        for (int i = 0; i < dimX; ++i)
        {
            printf("[%3d,%3d]: re=%9.4f, im=%9.4f\n",
                    j, i,
                    *(data + 2 * (j * dimX + i) + 0),
                    *(data + 2 * (j * dimX + i) + 1));
        }
}

///////////////////////////////////////////////////////////////////////

HD_CALLABLE ImageStream::ImageStream(Complex* base, int nCol, int nRow)
{
    initStream(base, nCol, nRow);
}

HD_CALLABLE void ImageStream::initStream(Complex* base, int nCol, int nRow)
{
    init(base, nCol, nRow);
    _baseFT = base;
}

HD_CALLABLE void ImageStream::indexImage(int index)
{
    _dataFT = _baseFT + _size * index;
}

///////////////////////////////////////////////////////////////////////

} // end namespace cuthunder

///////////////////////////////////////////////////////////////////////
