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

#include "Typedef.h"
#include "Enum.h"
#include "Error.h"
#include "Complex.h"

#include "ImageBase.h"
#include "BMP.h"
#include "Image.h"
#include "Interpolation.h"
#include "Functions.h"

#define VOLUME_CONJUGATE_HALF(iCol, iRow, iSlc) \
    (((iCol) > 0) ? 0 : [&iCol, &iRow, &iSlc]() \
                        { \
                            iCol *= -1; \
                            iRow *= -1; \
                            iSlc *= -1; \
                            return 1; \
                        }())

#define VOLUME_INDEX(i, j, k) \
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

#define VOLUME_FOR_EACH_PIXEL_RL(that) \
    for (int k = 0; k < that.nSlcRL(); k++) \
        for (int j = 0; j < that.nRowRL(); j++) \
            for (int i = 0; i < that.nColRL(); i++)

#define VOLUME_FOR_EACH_PIXEL_FT(that) \
    for (int k = -that.nSlcRL() / 2; k < that.nSlcRL() / 2; k++) \
        for (int j = -that.nRowRL() / 2; j < that.nSlcRL() / 2; j++) \
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
               const Space space);

        Volume(const Volume& that);

        ~Volume();
        
        Volume& operator=(const Volume& that);

        void alloc(const Space space);

        void alloc(const int nCol,
                   const int nRow,
                   const int nSlc,
                   const Space space);

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
