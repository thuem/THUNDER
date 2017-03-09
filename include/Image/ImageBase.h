/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef IMAGE_BASE
#define IMAGE_BASE

//#define CXX11_PTR
#define FFTW_PTR

#ifdef FFTW_PTR
#define FFTW_PTR_THREAD_SAFETY
#endif

#include <functional>
#include <cstring>
#include <cstdio>
#include <algorithm>

#include <fftw3.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#include <boost/move/core.hpp>
#include <boost/move/make_unique.hpp>

#include "Config.h"
#include "Macro.h"
#include "Complex.h"
#include "Typedef.h"
#include "Functions.h"
#include "Utils.h"
#include "Logging.h"

#define RL_SPACE 0

#define FT_SPACE 1

/**
 * This macro loops over each pixel of an Image / Volume in real space.
 * @param base an Image / Volume
 */
#define FOR_EACH_PIXEL_RL(base) \
    for (size_t i = 0; i < base.sizeRL(); i++)

/**
 * This macro loops over each pixel of an Image / Volume in Fourier space.
 * @param base an Image / Volume
 */
#define FOR_EACH_PIXEL_FT(base) \
    for (size_t i = 0; i < base.sizeFT(); i++)

/**
 * This macro sets each pixel of an Image / Volume to 0 in real space.
 * @param base an Image / Volume
 */
#define SET_0_RL(base) \
    FOR_EACH_PIXEL_RL(base) \
        base(i) = 0
    //memset(&base(0), 0, sizeof(double) * base.sizeRL());

/**
 * This macro sets each pixel of an Image / Volume to 0 in Fourier space.
 * @param base an Image / Volume
 */
#define SET_0_FT(base) \
    FOR_EACH_PIXEL_FT(base) \
        base[i] = COMPLEX(0, 0)
    //memset(&base[0], 0, sizeof(Complex) * base.sizeFT());

/**
 * This macro sets each pixel of an Image / Volume to 1 in real space.
 * @param base an Image / Volume
 */
#define SET_1_RL(base) \
    FOR_EACH_PIXEL_RL(base) \
        base(i) = 1

/**
 * This macro sets each pixel of an Image / Volume to 1 in Fourier space.
 * @param base an Image / Volume
 */
#define SET_1_FT(base) \
    FOR_EACH_PIXEL_FT(base) \
        base[i] = COMPLEX(1, 0)

/**
 * This macro negatives each pixel of an Image / Volume in real space.
 * @param base an Image / Volume
 */
#define NEG_RL(base) \
    SCALE_RL(base, -1)

/**
 * This macro negatives each pixel of an Image / Volume in Fourier space.
 * @param base an Image / Volume
 */
#define NEG_FT(base) \
    SCALE_FT(base, -1)

/**
 * This macro scales each pixel of an Image / Volume with a certain factor in
 * real space.
 * @param base an Image / Volume
 * @param a the scale factor
 */
#define SCALE_RL(base, a) \
    FOR_EACH_PIXEL_RL(base) \
        (base)(i) *= a

/**
 * This macro scales each pixel of an Image / Volume with a certain factor in
 * Fourier space.
 * @param base an Image / Volume
 * @param a the scale factor
 */
#define SCALE_FT(base, a) \
    FOR_EACH_PIXEL_FT(base) \
        (base)[i] *= a

#define COPY_RL(a, b) \
    FOR_EACH_PIXEL_RL(a) \
        (a)(i) = b(i)

#define COPY_FT(a, b) \
    FOR_EACH_PIXEL_FT(a) \
        (a)[i] = b[i]

/**
 * This macro adds each pixel from B to A respectively in real space.
 * @param a Image / Volume A
 * @param b Image / Volume B
 */
#define ADD_RL(a, b) \
    FOR_EACH_PIXEL_RL(a) \
        (a)(i) += (b)(i)

/**
 * This macro adds each pixel from B to A respectively in Fourier space.
 * @param a Image / Volume A
 * @param b Image / Volume B
 */
#define ADD_FT(a, b) \
    FOR_EACH_PIXEL_FT(a) \
        (a)[i] += (b)[i]

/**
 * This macro substracts each pixel from B to A respectively in real space.
 * @param a Image / Volume A
 * @param b Image / Volume B
 */
#define SUB_RL(a, b) \
    FOR_EACH_PIXEL_RL(a) \
        (a)(i) -= (b)(i)

/**
 * This macro substracts each pixel from B to A respectively in Fourier space.
 * @param a Image / Volume A
 * @param b Image / Volume B
 */
#define SUB_FT(a, b) \
    FOR_EACH_PIXEL_FT(a) \
        (a)[i] -= (b)[i]

#define MUL_RL(a, b) \
    FOR_EACH_PIXEL_RL(a) \
        (a)(i) *= (b)(i)

/**
 * This macro multiplys each pixel of A with the pixel of B respectively in Fourier space.
 * @param a Image / Volume A
 * @param b Image / Volume B
 */
#define MUL_FT(a, b) \
    FOR_EACH_PIXEL_FT(a) \
        (a)[i] *= (b)[i]

#define DIV_RL(a, b) \
    FOR_EACH_PIXEL_RL(a) \
        (a)(i) /= (b)(i)

/**
 * This macro divides each pixel of A with the pixel of B respectively in Fourier space.
 * @param a Image / Volume A
 * @param b Image / Volume B
 */
#define DIV_FT(a, b) \
    FOR_EACH_PIXEL_FT(a) \
        (a)[i] /= (b)[i]

#define REMOVE_NEG(base) \
    FOR_EACH_PIXEL_RL(base) \
        if (base(i) < 0) base(i) = 0

#define BOUNDARY_CHECK_RL(index) \
    if ((index < 0) || (index >= _sizeRL)) \
        REPORT_ERROR("OUT OF BOUNDARY RL");

#define BOUNDARY_CHECK_FT(index) \
    if ((index < 0) || (index >= _sizeFT)) \
        REPORT_ERROR("OUT OF BOUNDARY FT");

class ImageBase
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE(ImageBase)

    protected:

#ifdef CXX11_PTR
        boost::movelib::unique_ptr<double[]> _dataRL;

        boost::movelib::unique_ptr<Complex[]> _dataFT;
#endif

#ifdef FFTW_PTR
        double* _dataRL;

        Complex* _dataFT;
#endif

        size_t _sizeRL;

        size_t _sizeFT;

        ImageBase();

        ImageBase(BOOST_RV_REF(ImageBase) that) : _dataRL(boost::move(that._dataRL)),
                                                  _dataFT(boost::move(that._dataFT)),
                                                  _sizeRL(that._sizeRL),
                                                  _sizeFT(that._sizeFT)
        {
            that._sizeRL = 0;
            that._sizeFT = 0;

#ifdef FFTW_PTR
            that._dataRL = NULL;
            that._dataFT = NULL;
#endif
        }

        ~ImageBase();

    public:

        void swap(ImageBase& that);

        /**
         * return a const pointer which points to the i-th element in real space
         * @param i index of the element
         */
        inline const double& iGetRL(const size_t i = 0) const
        {
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
            BOUNDARY_CHECK_RL(i);
#endif

            return _dataRL[i];
        };

        /**
         * return a const pointer which points to the i-th element in Fourier
         * space
         * @param i index of the element
         */
        inline const Complex& iGetFT(const size_t i = 0) const
        {
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
            BOUNDARY_CHECK_FT(i);
#endif

            return _dataFT[i];
        };

        /**
         * return the i-th element in real space
         * @param i index of the element
         */
        inline double& operator()(const size_t i)
        {
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
            BOUNDARY_CHECK_RL(i);
#endif

            return _dataRL[i];
        };

        /**
         * return the i-th element in Fourier space
         * @param i index of the element
         */
        inline Complex& operator[](const size_t i)
        {
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
            BOUNDARY_CHECK_FT(i);
#endif

            return _dataFT[i];
        };

        /**
         * check whether _dataRL is NULL or not
         */
        bool isEmptyRL() const;

        /**
         * check whether _dataFT is nULL or not
         */
        bool isEmptyFT() const;

        /**
         * return the number of pixels in real space
         */
        size_t sizeRL() const;

        /**
         * return the number of pixels in Fourier space
         */
        size_t sizeFT() const;

        /**
         * free the allocated space both in real space and Fouier space
         */
        void clear();

        /**
         * free the allocated space in real space
         */
        void clearRL();

        /**
         * free the allocated space in Fourier space
         */
        void clearFT();

        void copyBase(ImageBase&) const;

        ImageBase copyBase() const;
};

double norm(ImageBase& base);

void normalise(ImageBase& base);

#endif // IMAGE_BASE
