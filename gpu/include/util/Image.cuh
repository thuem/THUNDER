/***********************************************************************
 * FileName: Image.cuh
 * Author  : Kunpeng WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#ifndef IMAGE_CUH
#define IMAGE_CUH

#include "Config.h"
#include "huabin.h"

#include "Device.cuh"
#include "Complex.cuh"

namespace cuthunder{

#define CU_IMAGE_INDEX_FT(i, j) \
    (j) * (_nCol / 2 + 1) + (i)

#define CU_IMAGE_FOR_EACH_PIXEL_FT(that, jRow, iCol) \
    for (int jRow = -that.nRowFT() / 2; jRow < that.nRowFT() / 2; jRow++) \
        for (int iCol = 0; iCol < that.nColFT(); iCol++)

struct CTFAttr
{
    RFLOAT voltage;

    RFLOAT defocusU;

    RFLOAT defocusV;

    RFLOAT defocusTheta;

    RFLOAT Cs;

    RFLOAT amplitudeContrast;

    RFLOAT phaseShift;
};

class Image
{
    public:

        HD_CALLABLE Image() {}

        HD_CALLABLE ~Image() {}

        HD_CALLABLE void init(Complex *data, int nCol, int nRow);

        HD_CALLABLE int nCol() const { return _nCol; }

        HD_CALLABLE int nRow() const { return _nRow; }

        HD_CALLABLE int nColFT() const { return (_nCol / 2 + 1); }

        HD_CALLABLE int nRowFT() const { return _nRow; }

        HD_CALLABLE void devPtr(Complex *data) { _dataFT = data; }
        
        HD_CALLABLE Complex* devPtr() const { return _dataFT; }

        HD_CALLABLE bool coordinatesInBoundaryFT(const int iCol,
                                                 const int iRow) const;

        HD_CALLABLE Complex getFT(const int iCol,
                                  const int iRow) const;

        HD_CALLABLE void setFT(Complex value,
                               const int iCol,
                               const int iRow);

        HD_CALLABLE void addFT(Complex value,
                               const int iCol,
                               const int iRow);

        HD_CALLABLE Complex* getFTData() const;

    private:

        HD_CALLABLE int getIndex(int i, int j, bool& flag) const;

    private:

        int _nCol = 0;
        int _nRow = 0;

    protected:

        int _size = 0;

        Complex *_dataFT = NULL;
    
};

class ImageStream : public Image
{
    public:

        HD_CALLABLE ImageStream() {}

        HD_CALLABLE ImageStream(Complex* base, int nCol, int nRow);

        HD_CALLABLE ~ImageStream() {}

        HD_CALLABLE void initStream(Complex* base, int nCol, int nRow);

        HD_CALLABLE void initBase() { _baseFT = _dataFT; }

        HD_CALLABLE void indexImage(int index);

        HD_CALLABLE Complex* devBasePtr() const { return _baseFT; } 

    private:

        Complex *_baseFT = NULL;
};

HD_CALLABLE void displayImageFT(Image& img);

} // end namaspace cuthunder

#endif
