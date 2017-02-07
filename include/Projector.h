/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef PROJECTOR_H
#define PROJECTOR_H

#include "Config.h"
#include "Macro.h"
#include "Complex.h"
#include "Logging.h"

#include "Euler.h"

#include "Image.h"
#include "Volume.h"

#include "Coordinate5D.h"

#include "ImageFunctions.h"

class Projector
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE(Projector)

    private:

        int _mode;

        /**
         * Only the signal beyond the max radius in frequnecy will be processed.
         * Max radius must be smaller than half of the shortest dimension of the
         * projectee. When projectee is set, max radius will be properly set as
         * well. However, it can be overwrited.
         */
        int _maxRadius;

        /**
         * the interpolation type (SINC_INTERP, LINEAR_INTERP, NEAREST_INTERP)
         */
        int _interp;

        /**
         * padding factor
         */
        int _pf;

        /**
         * the image to be projected
         */
        Image _projectee2D;

        /**
         * the volume to be projected
         */
        Volume _projectee3D;

    public:

        /**
         * default constructor
         */
        Projector();

        Projector(BOOST_RV_REF(Projector) that)
        {
            swap(that);
        }

        /**
         * default deconstructor
         */
        ~Projector();

        void swap(Projector& that);

        inline Projector& operator=(BOOST_RV_REF(Projector) that)
        {
            if (this != &that) swap(that);
            return *this;
        }

        /**
         * If there is an image to be projected, return false, otherwise return
         * true.
         */
        bool isEmpty2D() const;

        /**
         * If there is a volume to be projected, return false, otherwise return
         * true.
         */
        bool isEmpty3D() const;

        int mode() const;

        void setMode(const int mode);

        /**
         * This function returns the max radius for processing signal in Fourier
         * transform (in pixel).
         */
        int maxRadius() const;

        /**
         * This function sets the max radius for processing signal in Fourier
         * transform (in pixel).
         *
         * @param maxRadius the max radius
         */
        void setMaxRadius(const int maxRadius);

        /**
         * This function returns the interpolation type for this projection.
         */
        int interp() const;

        /**
         * This function sets the interpolation type for this projection.
         *
         * @param interp the interpolation type
         */
        void setInterp(const int interp);

        /**
         * This function returns the paddding factor.
         */
        int pf() const;

        /**
         * This function sets the padding factor.
         *
         * @param pf the padding factor
         */
        void setPf(const int pf);

        /**
         * This function returns a constant reference to the projectee.
         */
        const Image& projectee2D() const;

        /**
         * This function returns a constant reference to the projectee.
         */
        const Volume& projectee3D() const;

        /**
         * This function sets the projectee. Moreover, it automatically sets the
         * max radius of processing signal.
         *
         * @param src the image to be projected
         */
        void setProjectee(Image src);

        /**
         * This function sets the projectee. Moreover, it automatically sets the
         * max radius of processing signal.
         *
         * @param src the volume to be projected
         */
        void setProjectee(Volume src);

        void project(Image& dst,
                     const mat22& mat) const;

        /**
         * This function projects given the rotation matrix.
         * 
         * @param dst the destination image
         * @param mat the rotation matrix
         */
        void project(Image& dst,
                     const mat33& mat) const;

        void project(Image& dst,
                     const mat22& mat,
                     const int* iCol,
                     const int* iRow,
                     const int* iPxl,
                     const int nPxl) const;

        /**
         * This function projects given the rotation matrix and pre-determined
         * pixel indices.
         *
         * @param dst  the destination image
         * @param mat  the rotation matrix
         * @param iCol the index of column
         * @param iRow the index of row
         * @param iPxl the pixel index
         * @param nPxl the number of pixels
         */
        void project(Image& dst,
                     const mat33& mat,
                     const int* iCol,
                     const int* iRow,
                     const int* iPxl,
                     const int nPxl) const;

        void project(Complex* dst,
                     const mat22& mat,
                     const int* iCol,
                     const int* iRow,
                     const int nPxl) const;

        void project(Complex* dst,
                     const mat33& mat,
                     const int* iCol,
                     const int* iRow,
                     const int nPxl) const;

        void projectMT(Image& dst,
                       const mat22& mat) const;

        /**
         * This function projects given the rotation matrix using multiple
         * threads.
         *
         * @param dst the destination image
         * @param mat the rotation matrix
         */
        void projectMT(Image& dst,
                       const mat33& mat) const;

        void projectMT(Image& dst,
                       const mat22& mat,
                       const int* iCol,
                       const int* iRow,
                       const int* iPxl,
                       const int nPxl) const;

        /**
         * This function projects given the rotation matrix and pre-determined
         * pixel indices using multiple threads.
         *
         * @param dst  the destination image
         * @param mat  the rotation matrix
         * @param iCol the index of column
         * @param iRow the index of row
         * @param iPxl the pixel index
         * @param nPxl the number of pixels
         */
        void projectMT(Image& dst,
                       const mat33& mat,
                       const int* iCol,
                       const int* iRow,
                       const int* iPxl,
                       const int nPxl) const;

        void projectMT(Complex* dst,
                       const mat22& mat,
                       const int* iCol,
                       const int* iRow,
                       const int nPxl) const;

        void projectMT(Complex* dst,
                       const mat33& mat,
                       const int* iCol,
                       const int* iRow,
                       const int nPxl) const;

        void project(Image& dst,
                     const mat22& rot,
                     const vec2& t) const;
        /**
         * This function projects given a rotation matrix and a translation
         * vector.
         *
         * @param dst the destination image
         * @param rot the rotation matrix
         * @param t   the translation vector
         */
        void project(Image& dst,
                     const mat33& rot,
                     const vec2& t) const;

        void project(Image& dst,
                     const mat22& rot,
                     const vec2& t,
                     const int* iCol,
                     const int* iRow,
                     const int* iPxl,
                     const int nPxl) const;

        void project(Image& dst,
                     const mat33& rot,
                     const vec2& t,
                     const int* iCol,
                     const int* iRow,
                     const int* iPxl,
                     const int nPxl) const;

        void project(Complex* dst,
                     const mat22& rot,
                     const vec2& t,
                     const int nCol,
                     const int nRow,
                     const int* iCol,
                     const int* iRow,
                     const int nPxl) const;

        void project(Complex* dst,
                     const mat33& rot,
                     const vec2& t,
                     const int nCol,
                     const int nRow,
                     const int* iCol,
                     const int* iRow,
                     const int nPxl) const;

        void projectMT(Image& dst,
                       const mat22& rot,
                       const vec2& t) const;

        /**
         * This function projects given a rotation matrix and a translation
         * vector using multiple threads.
         *
         * @param dst the destination image
         * @param rot the rotation matrix
         * @param t   the translation vector
         */
        void projectMT(Image& dst,
                       const mat33& rot,
                       const vec2& t) const;

        void projectMT(Image& dst,
                       const mat22& rot,
                       const vec2& t,
                       const int* iCol,
                       const int* iRow,
                       const int* iPxl,
                       const int nPxl) const;

        void projectMT(Image& dst,
                       const mat33& rot,
                       const vec2& t,
                       const int* iCol,
                       const int* iRow,
                       const int* iPxl,
                       const int nPxl) const;

        void projectMT(Complex* dst,
                       const mat22& rot,
                       const vec2& t,
                       const int nCol,
                       const int nRow,
                       const int* iCol,
                       const int* iRow,
                       const int nPxl) const;

        void projectMT(Complex* dst,
                       const mat33& rot,
                       const vec2& t,
                       const int nCol,
                       const int nRow,
                       const int* iCol,
                       const int* iRow,
                       const int nPxl) const;

    private:

        /**
         * This function performs gridding correction on projectee.
         */
        void gridCorrection();
};

#endif // PROJECTOR_H
