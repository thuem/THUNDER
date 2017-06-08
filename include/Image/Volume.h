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

#include "omp_compat.h"

#include "Config.h"
#include "Macro.h"
#include "Typedef.h"
#include "Complex.h"
#include "Logging.h"

#include "ImageBase.h"
#include "BMP.h"
#include "Image.h"
#include "Interpolation.h"
#include "Functions.h"
#include "TabFunction.h"
#include "Coordinate5D.h"

/***
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
***/

#define VOLUME_SUB_SPHERE_FT(a) \
    for (int k = GSL_MAX_INT(-_nSlc / 2, FLOOR(iSlc - a)); \
             k <= GSL_MIN_INT(_nSlc / 2 - 1, CEIL(iSlc + a)); \
             k++) \
        for (int j = GSL_MAX_INT(-_nRow / 2, FLOOR(iRow - a)); \
                 j <= GSL_MIN_INT(_nRow / 2 - 1, CEIL(iRow + a)); \
                 j++) \
            for (int i = GSL_MAX_INT(-_nCol / 2, FLOOR(iCol - a)); \
                     i <= GSL_MIN_INT(_nCol / 2, CEIL(iCol + a)); \
                     i++)

#define VOLUME_SUB_SPHERE_RL(a) \
    for (int k = GSL_MAX_INT(-_nSlc / 2, FLOOR(iSlc - a)); \
             k <= GSL_MIN_INT(_nSlc / 2 - 1, CEIL(iSlc + a)); \
             k++) \
        for (int j = GSL_MAX_INT(-_nRow / 2, FLOOR(iRow - a)); \
                 j <= GSL_MIN_INT(_nRow / 2 - 1, CEIL(iRow + a)); \
                 j++) \
            for (int i = GSL_MAX_INT(-_nCol / 2, FLOOR(iCol - a)); \
                     i <= GSL_MIN_INT(_nCol / 2 - 1, CEIL(iCol + a)); \
                     i++)

/**
 * This macro loops over each pixel of a volume in real space.
 *
 * @param that the volume
 */
#define VOLUME_FOR_EACH_PIXEL_RL(that) \
    for (int k = -that.nSlcRL() / 2; k < that.nSlcRL() / 2; k++) \
        for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
            for (int i = -that.nColRL() / 2; i < that.nColRL() / 2; i++) \

/**
 * This macro loops over each pixel of a volume in Fourier space.
 *
 * @param that the volume
 */
#define VOLUME_FOR_EACH_PIXEL_FT(that) \
    for (int k = -that.nSlcRL() / 2; k < that.nSlcRL() / 2; k++) \
        for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
            for (int i = 0; i <= that.nColRL() / 2; i++)

#define VOLUME_FOR_PIXEL_R_RL(r) \
    for (int k = -r; k < r; k++) \
        for (int j = -r; j < r; j++) \
            for (int i = -r; i < r; i++)

/**
 * This macro loops over the pixels of an image in a certain radius in Fourier
 * space.
 *
 * @param r the radius
 */
#define VOLUME_FOR_PIXEL_R_FT(r) \
    for (int k = -r; k < r; k++) \
        for (int j = -r; j < r; j++) \
            for (int i = 0; i <= r; i++)

inline bool conjHalf(int& iCol,
                     int& iRow,
                     int& iSlc)
{
    if (iCol >= 0) return false;

    iCol *= -1;
    iRow *= -1;
    iSlc *= -1;

    return true;
};

inline bool conjHalf(double& iCol,
                     double& iRow,
                     double& iSlc)
{
    if (iCol >= 0) return false;

    iCol *= -1;
    iRow *= -1;
    iSlc *= -1;

    return true;
}

class Volume : public ImageBase
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE(Volume)

    public:

        /**
         * number of columns of this volume
         */
        int _nCol;

        /**
         * number of rows of this volume
         */
        int _nRow;

        /**
         * number of slices of this volume
         */
        int _nSlc;

        int _nColFT;

        size_t _box[2][2][2];

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
         * move constructor
         *
         * @param that the original volume
         */
        Volume(BOOST_RV_REF(Volume) that) : ImageBase(BOOST_MOVE_BASE(ImageBase, that)),
                                            _nCol(that._nCol),
                                            _nRow(that._nRow),
                                            _nSlc(that._nSlc)
        {
            _nColFT = that._nColFT;

            FOR_CELL_DIM_3
                _box[k][j][i] = that._box[k][j][i];

            that._nCol = 0;
            that._nRow = 0;
            that._nSlc = 0;
        }

        /**
         * The deconstructor will automatically free all allocated space.
         */
        ~Volume();

        inline Volume& operator=(BOOST_RV_REF(Volume) that)
        {
            if (this != &that) swap(that);
            return *this;
        }

        void swap(Volume& that);

        Volume copyVolume() const;

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
         * This function gets the value of the voxel in real space at a given
         * coordinate.
         *
         * @param iCol the index of the column of this voxel in real space
         * @param iRow the index of the row of this voxel in real space
         * @param iSlc the index of the slice of this voxel in real space
         */
        double getRL(const int iCol,
                     const int iRow,
                     const int iSlc) const;

        /**
         * This function sets the value of the voxel in real space at a given
         * coordinate.
         *
         * @param value the value of the voxel in real space
         * @param iCol the index of the column of this voxel in real space
         * @param iRow the index of the row of this voxel in real space
         * @param iSlc the index of the slice of this voxel in real space
         */
        void setRL(const double value,
                   const int iCol,
                   const int iRow,
                   const int iSlc);

        /**
         * This function add a certain value on the voxel in real space at a
         * given coordinate.
         *
         * @param value the value of the voxel in real space
         * @param iCol the index of the column of this voxel in real space
         * @param iRow the index of the row of this voxel in real space
         * @param iSlc the index of the slice of this voxel in real space
         */
        void addRL(const double value,
                   const int iCol,
                   const int iRow,
                   const int iSlc);

        /**
         * This function gets the value of the voxel in Fourier space at a given
         * coordinate.
         *
         * @param iCol the index of the column of this voxel in Fourier space
         * @param iRow the index of the row of this voxel in Fourier space
         * @param iSlc the index of the slice of this voxel in Fourier space
         */
        Complex getFT(int iCol,
                      int iRow,
                      int iSlc) const;

        /**
         * This function gets the value of the voxel in Fourier space at a given
         * cooridnate.
         *
         * @param iCol the index of the column of this voxel in Fourier space
         * @param iRow the index of the row of this voxel in Fourier space
         * @param iSlc the index of the slice of this voxel in Fourier space
         */
        Complex getFTHalf(const int iCol,
                          const int iRow,
                          const int iSlc) const;

        /**
         * This function sets the value of the voxel in Fourier space at a given
         * coordinate.
         *
         * @param value the value of voxel in Fourier space
         * @param iCol the index of the column of this voxel in Fourier space
         * @param iRow the index of the row of this voxel in Fourier space
         * @param iSlc the index of the slice of this voxel in real space
         */
        void setFT(const Complex value,
                   int iCol,
                   int iRow,
                   int iSlc);

        /**
         * This function sets the value of the voxel in Fourier space at a given
         * coordinate.
         *
         * @param value the value of the voxel in Fourier space
         * @param iCol  the index of the column of this voxel in Fourier space
         * @param iRow  the index of the row of this voxel in Fourier space
         * @param iSlc  the index of the slice of this voxel in real space
         */
        void setFTHalf(const Complex value,
                       const int iCol,
                       const int iRow,
                       const int iSlc);
        
        /**
         * This function adds a certain value on a voxel in Fourier space at a
         * given coordinate.
         *
         * @param value the value of the voxel
         * @param iCol  the index of the column of this voxel in Fourier space
         * @param iRow  the index of the row of this voxel in Fourier space
         * @param iSlc  the index of the slice of this voxel in real space
         */
        void addFT(const Complex value,
                   int iCol,
                   int iRow,
                   int iSlc);

        /**
         * This function adds a certain value on a voxel in Fourier space at a
         * given coordiante.
         *
         * @param value the value of value
         * @param iCol  the index of the column of this voxel in Fourier space
         * @param iRow  the index of the row of this voxel in Fourier space
         * @param iSlc  the index of the slice of this voxel in real space
         */
        void addFTHalf(const Complex value,
                       const int iCol,
                       const int iRow,
                       const int iSlc);

        /**
         * This function addes the real part on a voxel in Fourier space at a
         * given coordinate.
         *
         * @param value the real part of the voxel
         * @param iCol  the index of the column of this voxel in Fourier space
         * @param iRow  the index of the row of this voxel in Fourier space
         * @param iSlc  the index of the slice of this voxel in real space
         */
        void addFT(const double value,
                   int iCol,
                   int iRow,
                   int iSlc);

        /**
         * This function addes the real part on a voxel in Fourier space at a
         * given coordinate.
         *
         * @param value the real part of the voxel
         * @param iCol  the index of the column of this voxel in Fourier space
         * @param iRow  the index of the row of this voxel in Fourier space
         * @param iSlc  the index of the slice of this voxel in real space
         */
        void addFTHalf(const double value,
                       const int iCol,
                       const int iRow,
                       const int iSlc);

        /**
         * This function returns the value of an unregular voxel in speace spce
         * by interpolation.
         *
         * @param iCol   the index of the column of this unregular voxel in
         *               Fourier space
         * @param iRow   the index of the row of this unregular voxel in
         *               Fourier space
         * @param iSlc   the index of the slice of this unregular voxel in
         *               Fourier space
         * @param interp indicate the type of interpolation, where INTERP_NEAREST
         *               stands for the nearest point interpolation,
         *               INTERP_LINEAR stands for the trilinear interpolation
         *               and INTERP_SINC stands for the sinc interpolation
         */
        double getByInterpolationRL(const double iCol,
                                    const double iRow,
                                    const double iSlc,
                                    const int interp) const;

        /**
         * This function returns the value of an unregular voxel in Fourier
         * space by interpolation.
         *
         * @param iCol   the index of the column of this unregular voxel in
         *               real space
         * @param iRow   the index of the row of this unregular voxel in
         *               real space
         * @param iSlc   the index of the slice of this unregular voxel in
         *               real space
         * @param interp indicate the type of interpolation, where INTERP_NEAREST
         *               stands for the nearest point interpolation,
         *               INTERP_LINEAR stands for the trilinear interpolation
         *               and INTERP_SINC stands for the sinc interpolation
         */
        Complex getByInterpolationFT(double iCol,
                                     double iRow,
                                     double iSlc,
                                     const int interp) const;

        void addFT(const Complex value,
                   double iCol,
                   double iRow,
                   double iSlc);

        void addFT(const double value,
                   double iCol,
                   double iRow,
                   double iSlc);

        /**
         * This function adds a certain value on an unregular voxel in Fourier
         * space by a kernal of Modified Kaiser Bessel Function.
         *
         * @param value the value to be added
         * @param iCol the index of the column of this unregular voxel in
         *             real space
         * @param iRow the index of the row of this unregular voxel in
         *             real space
         * @param iSlc the index of the slice of this unregular voxel in
         *             real space
         * @param a the radius of Modified Kaiser Bessel Function
         * @param alpha the smooth factor of Modified Kaiser Bessel Function
         */
        void addFT(const Complex value,
                   const double iCol,
                   const double iRow,
                   const double iSlc,
                   const double a,
                   const double alpha);

        /**
         * This function adds a certain value on the real part of an unregular
         * voxel in Fourier space by a kernal of Modified Kaiser Bessel
         * Function.
         *
         * @param value the value to be added
         * @param iCol the index of the column of this unregular voxel in
         *             real space
         * @param iRow the index of the row of this unregular voxel in
         *             real space
         * @param iSlc the index of the slice of this unregular voxel in
         *             real space
         * @param a the radius of Modified Kaiser Bessel Function
         * @param alpha the smooth factor of Modified Kaiser Bessel Function
         */
        void addFT(const double value,
                   const double iCol,
                   const double iRow,
                   const double iSlc,
                   const double a,
                   const double alpha);

        /**
         * This function adds a certain value on an unregualr voxel in Fourier
         * space by a certain kernel.
         *
         * @param value the value to be added
         * @param iCol the index of the column of this unregular voxel in
         *             real space
         * @param iRow the index of the row of this unregular voxel in
         *             real space
         * @param iSlc the index of the slice of this unregular voxel in
         *             real space
         * @param a the radius of the blob
         * @param kernel a tabular function indicating the kernel which is a
         *               function of only one parameter, the square of radius
         */
        void addFT(const Complex value,
                   const double iCol,
                   const double iRow,
                   const double iSlc,
                   const double a,
                   const TabFunction& kernel);

        /**
         * This function adds a certain value on the real part of an unregualr
         * voxel in Fourier space by a certain kernel.
         *
         * @param value the value to be added
         * @param iCol the index of the column of this unregular voxel in
         *             real space
         * @param iRow the index of the row of this unregular voxel in
         *             real space
         * @param iSlc the index of the slice of this unregular voxel in
         *             real space
         * @param a the radius of the blob
         * @param kernel a tabular function indicating the kernel which is a
         *               function of only one paramter, the square of radius
         */
        void addFT(const double value,
                   const double iCol,
                   const double iRow,
                   const double iSlc,
                   const double a,
                   const TabFunction& kernel);

        /**
         * This function clears up the allocated space and resets the size of
         * the volume to 0.
         */
        void clear();

        inline size_t iRL(const int i,
                          const int j,
                          const int k) const
        {
            return (k >= 0 ? k : k + _nSlc) * _nCol * _nRow
                 + (j >= 0 ? j : j + _nRow) * _nCol
                 + (i >= 0 ? i : i + _nCol);
        }

        inline size_t iFT(int i,
                          int j,
                          int k) const
        {
            if (i >= 0)
                return iFTHalf(i, j, k);
            else
                return iFTHalf(-i, -j, -k);
        }

        inline size_t iFT(bool& conj,
                          int i,
                          int j,
                          int k) const
        {
            conj = conjHalf(i, j, k);

            return iFTHalf(i, j, k);
        }

        inline size_t iFTHalf(const int i,
                              const int j,
                              const int k) const
        {
            return (k >= 0 ? k : k + _nSlc) * _nColFT * _nRow
                 + (j >= 0 ? j : j + _nRow) * _nColFT 
                 + i;
        }

    private:

        void initBox();

        /**
         * This function checks whether the given coordinates is in the boundary
         * of the volume or not in real space. If not, it will crash the process
         * and record a fatal log.
         *
         * @param iCol the index of the column of this voxel in real space
         * @param iRow the index of the row of this voxel in real space
         * @param iSlc the index of the slice of this voxel in real space
         */
        void coordinatesInBoundaryRL(const int iCol,
                                     const int iRow,
                                     const int iSlc) const;

        /**
         * This function checks whether the given coordinates is in the boundary
         * of the volume or not in Fourier space. If not, it will crash the
         * process and record a fatal log.
         *
         * @param iCol the index of the column of this voxel in Fourier space
         * @param iRow the index of the row of this voxel in Fourier space
         * @param iSlc the index of the slice of this voxel in Fourier space
         */
        void coordinatesInBoundaryFT(const int iCol,
                                     const int iRow,
                                     const int iSlc) const;

        double getRL(const double w[2][2][2],
                     const int x0[3]) const;

        Complex getFTHalf(const double w[2][2][2],
                          const int x0[3]) const;

        void addFTHalf(const Complex value,
                       const double w[2][2][2],
                       const int x0[3]);

        void addFTHalf(const double value,
                       const double w[2][2][2],
                       const int x0[3]);
};

#endif // VOLUME_H
