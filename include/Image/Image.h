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

#include "Config.h"
#include "Macro.h"
#include "Typedef.h"
#include "Complex.h"
#include "Logging.h"

#include "Interpolation.h"

#include "ImageBase.h"
#include "BMP.h"

/**
 * This macro loops over each pixel of an image in real space.
 *
 * @param that the image
 */
#define IMAGE_FOR_EACH_PIXEL_RL(that) \
    for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
        for (int i = -that.nColRL() / 2; i < that.nColRL() / 2; i++)

/**
 * This macro loops over each pixel of an image in Fourier space.
 *
 * @param that the image
 */
#define IMAGE_FOR_EACH_PIXEL_FT(that) \
    for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
        for (int i = 0; i <= that.nColRL() / 2; i++)

/**
 * This macro loops over the pixels of an image in a certain radius in real
 * space.
 *
 * @param r the radius
 */
#define IMAGE_FOR_PIXEL_R_RL(r) \
    for (int j = -r; j < r; j++) \
        for (int i = -r; i < r; i++)

/**
 * This macro loops over the pixels of an image in a certain radius in Fourier
 * space.
 *
 * @param r the radius
 */
#define IMAGE_FOR_PIXEL_R_FT(r) \
    for (int j = -r; j < r; j++) \
        for (int i = 0; i <= r; i++)

inline bool conjHalf(int& iCol,
                     int& iRow)
{
    if (iCol >= 0) return false;

    iCol *= -1;
    iRow *= -1;

    return true;
};

inline bool conjHalf(double& iCol,
                     double& iRow)
{
    if (iCol >= 0) return false;

    iCol *= -1;
    iRow *= -1;

    return true;
}

class Image : public ImageBase
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE(Image)

    protected:

        /**
         * number of columns
         */
        int _nCol;

        /**
         * number of rows
         */
        int _nRow;

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

        Image(BOOST_RV_REF(Image) that) : ImageBase(BOOST_MOVE_BASE(ImageBase, that)),
                                          _nCol(that._nCol),
                                          _nRow(that._nRow)
        {
            that._nCol = 0;
            that._nRow = 0;
        }

        /**
         * deconstructor
         */
        ~Image();

        inline Image& operator=(BOOST_RV_REF(Image) that)
        {
            if (this != &that) swap(that);
            return *this;
        }

        void swap(Image& that);

        Image copyImage() const;

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
         *
         * @param iCol index of the column
         * @param iRow index of the row
         */
        Complex getFT(int iCol,
                      int iRow) const;

        Complex getFTHalf(const int iCol,
                          const int iRow) const;

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

        void setFTHalf(const Complex value,
                       const int iCol,
                       const int iRow);

        void addFT(const Complex value,
                   int iCol,
                   int iRow);

        void addFTHalf(const Complex value,
                       const int iCol,
                       const int iRow);

        void addFT(const double value,
                   int iCol,
                   int iRow);

        void addFTHalf(const double value,
                       const int iCol,
                       const int iRow);

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

        Complex getByInterpolationFT(double iCol,
                                     double iRow,
                                     const int interp) const;

        void addFT(const Complex value,
                   double iCol,
                   double iRow);

        void addFT(const double value,
                   double iCol,
                   double iRow);

        void clear()
        {
            ImageBase::clear();
            _nRow = 0;
            _nCol = 0;
        }

        inline int iRL(const int i,
                       const int j) const
        {
            return (j >= 0 ? j : j + _nRow) * _nCol
                 + (i >= 0 ? i : i + _nCol);
        }

        inline int iFT(int i,
                       int j) const
        {
            if (i >= 0)
                return iFTHalf(i, j);
            else
                return iFTHalf(-i, -j);
        }

        inline int iFT(bool& conj,
                       int i,
                       int j) const
        {
            conj = conjHalf(i, j);

            return iFTHalf(i, j);
        }

        inline int iFTHalf(const int i,
                           const int j) const
        {
            return (j >= 0 ? j : j + _nRow) * (_nCol / 2 + 1) + i;
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

        Complex getFTHalf(const double w[2][2],
                          const int x0[2]) const;

        void addFTHalf(const Complex value,
                       const double w[2][2],
                       const int x0[2]);

        void addFTHalf(const double value,
                       const double w[2][2],
                       const int x0[2]);
};

#endif // IMAGE_H
