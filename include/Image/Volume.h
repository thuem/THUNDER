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
#include <armadillo>

#include <omp.h>

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
    public:
    
        int _nCol = 0;
        int _nRow = 0;
        int _nSlc = 0;

    public:
        
        Volume();

        Volume(const int nCol,
               const int nRow,
               const int nSlc,
               const int space);

        Volume(const Volume& that);

        ~Volume();
        
        Volume& operator=(const Volume& that);

        void alloc(const int space);

        void alloc(const int nCol,
                   const int nRow,
                   const int nSlc,
                   const int space);

        int nColRL() const;
        int nRowRL() const;
        int nSlcRL() const;

        int nColFT() const;
        int nRowFT() const;
        int nSlcFT() const;

        double getRL(const int iCol,
                     const int iRow,
                     const int iSlc) const;

        void setRL(const double value,
                   const int iCol,
                   const int iRow,
                   const int iSlc);

        void addRL(const double value,
                   const int iCol,
                   const int iRow,
                   const int iSlc);

        Complex getFT(const int iCol,
                      const int iRow,
                      const int iSlc,
                      const ConjugateFlag cf = conjugateUnknown) const;
        // get the value of the Fourier volume according to given coordinates
        // the coordinates refer to the frequency information; not the way the
        // data actually output by FFTW and stored

        void setFT(const Complex value,
                   int iCol,
                   int iRow,
                   int iSlc,
                   const ConjugateFlag cf = conjugateUnknown);
        // set the value of the Fourier volume according to given coordinates
        // the coordinates refer to the frequency information; not the way the
        // data actually output by FFTW and stored

        void addFT(const Complex value,
                   int iCol,
                   int iRow,
                   int iSlc,
                   const ConjugateFlag cf = conjugateUnknown);
        // add the value of the Fourier volume according to given coordinates
        // the coordinates refer to the frequency information; not the way the
        // data actually output by FFTW and stored

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

        void addImages(std::vector<Image>& images,
                       std::vector<Coordinate5D>& coords,
                       const double maxRadius,
                       const double a,
                       const TabFunction& kernel);
        /* add whole images once */

        void addImage(const int iCol,
                      const int iRow,
                      const int iSlc,
                      const Image& image,
                      const arma::mat33& mat,
                      const TabFunction& kernel,
                      const double w = 1.0,
                      const double a = 1.9,
                      const int _pf = 2);
        /* add a image once */

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
