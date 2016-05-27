/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef VOLUME_H
#define VOLUME_H

#include <cmath>
#include <cstdlib>
#include <vector>

#include "omp_if.h"
#include "Typedef.h"
#include "Enum.h"
#include "Error.h"
#include "Complex.h"

#include "ImageBase.h"
#include "BMP.h"
#include "Image.h"
#include "Interpolation.h"
#include "Functions.h"
#include "TabFunction.h"
#include "Coordinate5D.h"
#include "Transformation.h"
#include "ImageFunctions.h"

#define VOLUME_CONJUGATE_HALF(iCol, iRow, iSlc) \
    (((iCol) >= 0) ? 0 : [&iCol, &iRow, &iSlc]() \
                         { \
                             iCol *= -1; \
                             iRow *= -1; \
                             iSlc *= -1; \
                             return 1; \
                         }())

#define VOLUME_INDEX_RL(i, j, k) \
    (k) * _nRow * _nCol + (j) * _nCol + (i)

#define VOLUME_INDEX_FT(i, j, k) \
    (k) * _nRow * (_nCol / 2 + 1) + (j) * (_nCol / 2 + 1) + (i)

#define VOLUME_FREQ_TO_STORE_INDEX_HALF(i, j, k) \
    [this](int ii, int jj, int kk) \
    { \
        if ((jj) < 0) jj += _nRow; \
        if ((kk) < 0) kk += _nSlc; \
        return VOLUME_INDEX_FT(ii, jj, kk); \
    }(i, j, k)

#define VOLUME_FREQ_TO_STORE_INDEX(index, flag, i, j, k, cf) \
    [this, &index, &flag, i, j, k, cf]() mutable \
    { \
        switch (cf) \
        { \
            case conjugateUnknown: \
                flag = VOLUME_CONJUGATE_HALF(i, j, k); break; \
            case conjugateYes: \
                flag = true; break; \
            case conjugateNo: \
                flag = false; break; \
        } \
        index = VOLUME_FREQ_TO_STORE_INDEX_HALF(i, j, k); \
    }()

#define VOLUME_SUB_SPHERE_FT(a) \
    for (int k = MAX(-_nSlc / 2, floor(iSlc - a)); \
             k <= MIN(_nSlc / 2 - 1, ceil(iSlc + a)); \
             k++) \
        for (int j = MAX(-_nRow / 2, floor(iRow - a)); \
                 j <= MIN(_nRow / 2 - 1, ceil(iRow + a)); \
                 j++) \
            for (int i = MAX(-_nCol / 2, floor(iCol - a)); \
                     i <= MIN(_nCol / 2, ceil(iCol + a)); \
                     i++)

#define VOLUME_FOR_EACH_PIXEL_RL(that) \
    for (int k = -that.nSlcRL() / 2; k < that.nSlcRL() / 2; k++) \
        for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
            for (int i = -that.nColRL() / 2; i < that.nColRL() / 2; i++) \

#define VOLUME_FOR_EACH_PIXEL_FT(that) \
    for (int k = -that.nSlcRL() / 2; k < that.nSlcRL() / 2; k++) \
        for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
            for (int i = 0; i <= that.nColRL() / 2; i++)

class Volume : public ImageBase
{
    MAKE_DEFAULT_MOVE(Volume)

    public:

        /**
         * number of columns of this volume
         */
        int _nCol = 0;

        /**
         * number of rows of this volume
         */
        int _nRow = 0;

        /**
         * number of slices of this volume
         */
        int _nSlc = 0;

    public:

        /**
         * default constructor
         */
        Volume();

        /**
         * constructor
         *
         * @param nCol number of columns of this volume
         * @param nRow number of rows of this volume
         * @param nSlc number of slices of this volume
         * @param space the space this volume allocating, where RL_SPACE stands
         *              for the real space and FT_SPACE stands for the Fourier
         *              space
         */
        Volume(const int nCol,
               const int nRow,
               const int nSlc,
               const int space);

        /**
         * The deconstructor will automatically free all allocated space.
         */
        ~Volume();

        /**
         * This function allocates a piece of memory in a certain space.
         * @param space the space this volume allocating, where RL_SPACE stands
         *              for the real space and FT_SPACE stands for the Fourier
         *              space
         */
        void alloc(const int space);

        /* This function allocates a piece of memory in a certain space.
         *
         * @param nCol number of columns of this volume
         * @param nRow number of rows of this volume
         * @param nSlc number of slices of this volume
         * @param space the space this volume allocating, where RL_SPACE stands
         *              for the real space and FT_SPACE stands for the Fourier
         *              space
         */
        void alloc(const int nCol,
                   const int nRow,
                   const int nSlc,
                   const int space);

        /**
         * This function returns the number of columns of this volume in real
         * space.
         */
        inline int nColRL() const { return _nCol; };

        /**
         * This function returns the number of rows of this volume in real
         * space.
         */
        inline int nRowRL() const { return _nRow; };

        /**
         * This function returns the number of rows of this volume in real
         * space.
         */
        inline int nSlcRL() const { return _nSlc; };

        /**
         * This function returns the number of columns of this volume in Fourier
         * space.
         */
        inline int nColFT() const { return _nCol / 2 + 1; };

        /**
         * This function returns the number of rows of this volume in Fourier
         * space.
         */
        inline int nRowFT() const { return _nRow; };

        /**
         * This function returns the number of slices of this volume in Fourier
         * space.
         */
        inline int nSlcFT() const { return _nSlc; };

        /**
         * This function gets the value of the pixel in real space at a given
         * coordinate.
         *
         * @param iCol the index of the column of this pixel in real space
         * @param iRow the index of the row of this pixel in real space
         * @param iSlc the index of the slice of this pixel in real space
         */
        double getRL(const int iCol,
                     const int iRow,
                     const int iSlc) const;

        /**
         * This function sets the value of the pixel in real space at a given
         * coordinate.
         *
         * @param iCol the index of the column of this pixel in real space
         * @param iRow the index of the row of this pixel in real space
         * @param iSlc the index of the slice of this pixel in real space
         */
        void setRL(const double value,
                   const int iCol,
                   const int iRow,
                   const int iSlc);

        /**
         * This function sets the value of the pixel in real space at a given
         * coordinate.
         *
         * @param iCol the index of the column of this pixel in real space
         * @param iRow the index of the row of this pixel in real space
         * @param iSlc the index of the slice of this pixel in real space
         */
        void addRL(const double value,
                   const int iCol,
                   const int iRow,
                   const int iSlc);

        /**
         * This function gets the value of the pixel in Fourier space at a given
         * coordinate.
         *
         * @param iCol the index of the column of this pixel in Fourier space
         * @param iRow the index of the row of this pixel in Fourier space
         * @param iSlc the index of the slice of this pixel in real space
         * @param cf the conjugate flag, where conjugateUnknown stands for
         * calculating the conjugate status on its own, conjugateYes stands
         * for returning the conjugate of the value of the pixel and conjugateNo
         * stands for returning the value of the pixel without conjugation
         */
        Complex getFT(const int iCol,
                      const int iRow,
                      const int iSlc,
                      const ConjugateFlag cf = conjugateUnknown) const;

        /**
         * This function sets the value of the pixel in Fourier space at a given
         * coordinate.
         *
         * @param iCol the index of the column of this pixel in Fourier space
         * @param iRow the index of the row of this pixel in Fourier space
         * @param iSlc the index of the slice of this pixel in real space
         * @param cf the conjugate flag, where conjugateUnknown stands for
         * calculating the conjugate status on its own, conjugateYes stands
         * for setting the conjugate of the value of the pixel and conjugateNo
         * stands for setting the value of the pixel without conjugation
         */
        void setFT(const Complex value,
                   int iCol,
                   int iRow,
                   int iSlc,
                   const ConjugateFlag cf = conjugateUnknown);
        
        /**
         * This function adds the value of the pixel in Fourier space at a given
         * coordinate.
         *
         * @param iCol the index of the column of this pixel in Fourier space
         * @param iRow the index of the row of this pixel in Fourier space
         * @param iSlc the index of the slice of this pixel in real space
         * @param cf the conjugate flag, where conjugateUnknown stands for
         * calculating the conjugate status on its own, conjugateYes stands
         * for adding the conjugate of the value of the pixel and conjugateNo
         * stands for adding the value of the pixel without conjugation
         */
        void addFT(const Complex value,
                   int iCol,
                   int iRow,
                   int iSlc,
                   const ConjugateFlag cf = conjugateUnknown);

        double getByInterpolationRL(const double iCol,
                                    const double iRow,
                                    const double iSlc,
                                    const int interp) const;

        Complex getByInterpolationFT(double iCol,
                                     double iRow,
                                     double iSlc,
                                     const int interp) const;

        void addFT(const Complex value,
                   const double iCol,
                   const double iRow,
                   const double iSlc,
                   const double a,
                   const double alpha);
        /* add by a kernel of Mofidied Kaiser Bessel Function */

        void addFT(const Complex value,
                   const double iCol,
                   const double iRow,
                   const double iSlc,
                   const double a,
                   const TabFunction& kernel);
        /* add by a given kernel */

        void clear()
        {
            ImageBase::clear();
            _nCol = 0;
            _nRow = 0;
            _nSlc = 0;
        }

        Volume copyVolume() const
        {
            Volume out;
            copyBase(out);
            out._nRow = _nRow;
            out._nCol = _nCol;
            out._nSlc = _nSlc;
            return out;
        }

    private:

        void coordinatesInBoundaryRL(const int iCol,
                                     const int iRow,
                                     const int iSlc) const;
        // check whether the input coordinates can in boundary of the volume
        // If not, throw out an Error

        void coordinatesInBoundaryFT(const int iCol,
                                     const int iRow,
                                     const int iSlc) const;
        // check whether the input coordinates can in boundary of the Fourier
        // volume
        // If not, throw out an Error

        double getRL(const double w[2][2][2],
                     const int x0[3]) const;

        Complex getFT(const double w[2][2][2],
                      const int x0[3],
                      const ConjugateFlag conjugateFlag) const;
};

#endif // VOLUME_H
