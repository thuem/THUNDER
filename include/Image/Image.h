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
#include "Logging.h"

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

/**
 * This macro loops over each pixel of an image in real space.
 * @param that the image
 */
#define IMAGE_FOR_EACH_PIXEL_RL(that) \
    for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
        for (int i = -that.nColRL() / 2; i < that.nColRL() / 2; i++)

/**
 * This macro loops over each pixel of an image in Fourier space.
 * @param that the image
 */
#define IMAGE_FOR_EACH_PIXEL_FT(that) \
    for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
        for (int i = 0; i <= that.nColRL() / 2; i++)

/**
 * This macro loops over the pixels of an image in a certain radius in real
 * space.
 * @param r the radius
 */
#define IMAGE_FOR_PIXEL_R_RL(r) \
    for (int j = -r; j < r; j++) \
        for (int i = -r; i < r; i++)

/**
 * This macro loops over the pixels of an image in a certain radius in Fourier
 * space.
 * @param r the radius
 */
#define IMAGE_FOR_PIXEL_R_FT(r) \
    for (int j = -r; j < r; j++) \
        for (int i = 0; i<= r; i++)

class Image : public ImageBase
{
    MAKE_DEFAULT_MOVE(Image)

    protected:

        /**
         * number of columns
         */
        int _nCol = 0;

        /**
         * number of rows
         */
        int _nRow = 0;

    public:

        /**
         * default constructor
         */
        Image();

        /**
         * This function is a constructor of Image. It constructs an Image
         * object with the given number of columns and number of row in certian
         * space.
         * @param nCol number of columns
         * @param nRow number of rows
         * @param space the space (RL_SPACE: real space, FT: Fourier space)
         */
        Image(const int nCol,
              const int nRow,
              const int space);

        /**
         * deconstructor
         */
        ~Image();

        /**
         * This function allocates memory space in a certain space.
         * @param space the space (RL_SPACE: real space, FT: Fourier space)
         */
        void alloc(const int space);

        /**
         * This function allocates memory space in a certain space with given
         * number of columns and number of rows.
         * @param nCol number of columns
         * @param nRow number of rows
         * @param space the space (RL_SPACE: real space, FT: Fourier space)
         */
        void alloc(const int nCol,
                   const int nRow,
                   const int space);

        /**
         * This function returns the number of columns in real space.
         */
        inline int nColRL() const { return _nCol; };

        /**
         * This function returns the number of rows in real space.
         */
        inline int nRowRL() const { return _nRow; };

        /**
         * This function returns the number of columns in Fourier space.
         */
        inline int nColFT() const { return _nCol / 2 + 1; };

        /**
         * This function returns the number of rows in Fourier space.
         */
        inline int nRowFT() const { return _nRow; };

        /**
         * This function saves the real space image to a BMP image. If the file
         * does not exist, create it.
         * @param filename the file name of the BMP image
         */
        void saveRLToBMP(const char* filename) const;

        /**
         * This function saves the Fourier space image to a BMP image. If the
         * file does not exist, create it.
         * @param filename the file name of the BMP image
         * @param c log(1 + mod(x) * c)
         */
        void saveFTToBMP(const char* filename,
                         const double c) const;

        /**
         * This function returns the value of the pixel at the certain column and
         * row in real space.
         * @param iCol index of the column
         * @param iRow index of the row
         */
        double getRL(const int iCol,
                     const int iRow) const;

        /**
         * This function sets the value of the pixel at the certain column and
         * row.
         * @param value the value the pixel to be set to
         * @param iCol index of the column
         * @param iRow index of the row
         */
        void setRL(const double value,
                   const int iCol,
                   const int iRow);

        /**
         * This function returns the value of the pixel at the certain column
         * and row in Fourier space.
         * @param iCol index of the column
         * @param iRow index of the row
         */
        Complex getFT(int iCol,
                      int iRow) const;

        /**
         * This function sets the value of the pixel at the certain column and
         * row in Fourier space.
         * @param value the value
         * @param iCol index of the column
         * @param iRow index of the row
         */
        void setFT(const Complex value,
                   int iCol,
                   int iRow);

        /**
         * This function gets the value of an irregular pixel by bi-linear
         * interpolation in real space.
         * @param iCol index of the column (irregular)
         * @param iRow index of the row (irregular)
         */
        double getBiLinearRL(const double iCol,
                             const double iRow) const;

        /**
         * This function gets the value of an irregular pixel by bi-linear
         * interpolation in Fourier space.
         * @param iCol index of the column (irregular)
         * @param iRow index of the row (irregular)
         */
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

        /**
         * This function checks whether the given coordinates is in the boundary
         * of the image or not in real space. If not, it will crash the process
         * and record a fatal log.
         *
         * @param iCol the index of the column of this pixel in real space
         * @param iRow the index of the row of this pixel in real space
         */
        void coordinatesInBoundaryRL(const int iCol,
                                     const int iRow) const;

        /**
         * This function checks whether the given coordinates is in the boundary
         * of the image or not in Fourier space. If not, it will crash the 
         * process and record a fatal log.
         *
         * @param iCol the index of the column of this pixel in real space
         * @param iRow the index of the row of this pixel in real space
         */
        void coordinatesInBoundaryFT(const int iCol,
                                     const int iRow) const;
};

#endif // IMAGE_H
