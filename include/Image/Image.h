/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef IMAGE_H
#define IMAGE_H

#include <cmath>
#include <cstdlib>

#include "Typedef.h"
#include "Enum.h"
#include "Error.h"
#include "Complex.h"

#include "Interpolation.h"

#include "ImageBase.h"
#include "BMP.h"

#define IMAGE_CONJUGATE_HALF(iCol, iRow) \
    (((iCol) >= 0) ? 0 : [&iCol, &iRow]() \
                         { \
                             iCol *= -1; \
                             iRow *= -1; \
                             return 1; \
                         }())

#define IMAGE_INDEX_RL(i, j) \
    (j) * _nCol + (i)

#define IMAGE_INDEX_FT(i, j) \
    (j) * (_nCol / 2 + 1) + (i)

#define IMAGE_FREQ_TO_STORE_INDEX(index, flag, i, j) \
    [this, &index, &flag, i, j]() mutable \
    { \
        flag = IMAGE_CONJUGATE_HALF(i, j); \
        if ((j) < 0) j += _nRow; \
        index = IMAGE_INDEX_FT(i, j); \
    }()

#define IMAGE_FOR_EACH_PIXEL_RL(that) \
    for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
        for (int i = -that.nColRL() / 2; i < that.nColRL() / 2; i++)

#define IMAGE_FOR_EACH_PIXEL_FT(that) \
    for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
        for (int i = 0; i <= that.nColRL() / 2; i++)

#define IMAGE_FOR_PIXEL_R_RL(r) \
    for (int j = -r; j < r; j++) \
        for (int i = -r; i < r; i++)

#define IMAGE_FOR_PIXEL_R_FT(r) \
    for (int j = -r; j < r; j++) \
        for (int i = 0; i<= r; i++)

class Image : public ImageBase
{
    MAKE_DEFAULT_MOVE(Image)

    protected:

        int _nCol = 0;
        int _nRow = 0;

    public:

        Image();

        Image(const int nCol,
              const int nRow,
              const int space);

        ~Image();

        void alloc(const int space);

        void alloc(const int nCol,
                   const int nRow,
                   const int space);

        int nColRL() const;
        int nRowRL() const;

        int nColFT() const;
        int nRowFT() const;

        void saveRLToBMP(const char* filename) const;
        /* Save _data to a bmp file. If the file does not exist, create it. */

        void saveFTToBMP(const char* filename,
                         const double c) const;
        /* Save _dataFT to a bmp file. If the file does not exist, create it.
         * log(1 + mod(x) * c) */

        double getRL(const int iCol,
                     const int iRow) const;

        void setRL(const double value,
                   const int iCol,
                   const int iRow);

        Complex getFT(int iCol,
                      int iRow) const;

        void setFT(const Complex value,
                   int iCol,
                   int iRow);

        double getBiLinearRL(const double iCol,
                             const double iRow) const;

        Complex getBiLinearFT(const double iCol,
                              const double iRow) const;

        void clear()
        {
            ImageBase::clear();
            _nRow = 0;
            _nCol = 0;
        }

        Image copyImage() const
        {
            Image out;
            copyBase(out);
            out._nRow = _nRow;
            out._nCol = _nCol;
            return out;
        }

    private:

        void coordinatesInBoundaryRL(const int iCol,
                                     const int iRow) const;
        // check whether the given coordinates are in the boundary of the image
        // if not, throw out an Error

        void coordinatesInBoundaryFT(const int iCol,
                                     const int iRow) const;
        // check whether the given coordinates are in the boundary of the
        // Fourier image.
        // if not, throw out an Error
};

#endif // IMAGE_H
