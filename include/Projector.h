/** @file
 *  @author Mingxu Hu
 *  @author Xiao Long
 *  @version 1.4.11.080920
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR    | TIME       | VERSION       | DESCRIPTION
 *  ------    | ----       | -------       | -----------
 *  Mingxu Hu | 2015/03/23 | 0.0.1.050323  | new file
 *  Xiao Long | 2018/09/20 | 1.4.11.080920 | add documentation
 *
 *  @brief Projector.h contains several functions for projecting images and volumes, given the rotation matrix, the pre-determined pixel indices and the translation vector. Moreover, same functions are re-accomplished using multiple threads.
 *
 * Note that, the given parameters determine the projection pattern.
 * 1. Project an image @f$I@f$, given the rotation matrix @f$\mathbf{M}@f$ and the padding factor @f$pf@f$, in Fourier space by interpolation, and output the projected image @f$I'@f$. For each pixel @f$\begin{pmatrix} x \\ y \end{pmatrix}@f$ of image @f$I@f$ in Fourier space, it will be projected to
 * \f[
 *   \begin{pmatrix} 
 *       x' \\
 *       y' \\
 *   \end{pmatrix}
 *   = M
 *   \begin{pmatrix}
 *       x * pf \\
 *       y * pf \\
 *   \end{pmatrix}
 * \f]
 * i.e. the value of projected image @f$I'@f$ pixel @f$\begin{pmatrix} x \\ y \end{pmatrix}@f$ in Fourier space is obtained by interpolation of original image @f$I@f$ pixel @f$\begin{pmatrix} x' \\ y' \end{pmatrix}@f$ in Fourier space.
 *
 * 2. Project a volume @f$V@f$, given the rotation matrix @f$\mathbf{M}@f$ and the padding factor @f$pf@f$, in Fourier space by interpolation, and output the projected image @f$I'@f$. For each voxel @f$\begin{pmatrix} x \\ y \\ z \end{pmatrix}@f$ of volume @f$V@f$ in Fourier space, it will be projected to
 * \f[
 *   \begin{pmatrix} 
 *       x' \\
 *       y' \\
 *       z' \\
 *   \end{pmatrix}
 *   = M
 *   \begin{pmatrix}
 *       x * pf \\
 *       y * pf \\
 *       0 \\
 *   \end{pmatrix}
 * \f]
 * i.e. the value of projected image @f$I'@f$ pixel @f$\begin{pmatrix} x \\ y \end{pmatrix}@f$ in Fourier space is obtained by interpolation of original volume @f$V@f$ voxel @f$\begin{pmatrix} x' \\ y' \\ z' \end{pmatrix}@f$ in Fourier space.
 *
 * 3. The pre-determined pixel indices @f$\mathbf{P} = \left\{\mathbf{p_0}, \mathbf{p_1}, \dots, \mathbf{p_{N - 1}}\right\}@f$ determine each pixel of projected image @f$I'@f$ in Fourier space. Given the column indices @f$\mathbf{C} = \left\{\mathbf{c_0}, \mathbf{c_1}, \dots, \mathbf{c_{N - 1}}\right\}@f$ and the row indices @f$\mathbf{R} = \left\{\mathbf{r_0}, \mathbf{r_1}, \dots, \mathbf{r_{N - 1}}\right\}@f$, for the @f$i@f$th pixel @f$p_i, 0 \leq i \leq N - 1@f$, of projected image @f$I'@f$, its coordinate in Fourier space is
 * \f[
 *   \begin{pmatrix} 
 *       x \\
 *       y \\
 *   \end{pmatrix}
 *   = 
 *   \begin{pmatrix}
 *       c_i \\
 *       r_i \\
 *   \end{pmatrix}
 * \f]
 *
 * 4. The translation vector is used in further translation after interpolation.
 */

#ifndef PROJECTOR_H
#define PROJECTOR_H

#include "Config.h"
#include "Macro.h"
#include "Complex.h"
#include "Logging.h"
#include "Precision.h"

#include "Euler.h"

#include "Image.h"
#include "Volume.h"

#include "Coordinate5D.h"

#include "ImageFunctions.h"

/**
 * @brief Class Projector defines attributes and functions used in projection.
 *
 * This class is movable but not copyable.
 */
class Projector
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE(Projector)

    private:

        int _mode;            /**< the mode, including MODE_2D and MODE_3D */

        int _maxRadius;       /**< the max radius. Only the signal within the max radius in frequency will be processed. Max radius must be smaller than half of the shortest dimension of the projectee. When projectee is set, max radius will be properly set as well. However, it can be overwritten. */

        int _interp;          /**< the interpolation type, including SINC_INTERP, LINEAR_INTERP and NEAREST_INTERP */

        int _pf;              /**< the padding factor, to expand the interpolation range */

        Image _projectee2D;   /**< the image to be projected */

        Volume _projectee3D;  /**< the volume to be projected */

    public:

        /**
         * @brief Default constructor.
         * 
         * Construct a new Projector class object, by initializing several key class data members.
         */
        Projector();

        /**
         * @brief Move constructor.
         * 
         * Construct a new Projector class object, by assigning all class data members from the source object to the constructed object.
         */
        Projector(BOOST_RV_REF(Projector) that  /**<[in] the source Projector */)
        {
            swap(that);
        }

        /**
         * @brief Default destructor.
         */
        ~Projector();

        /**
         * @brief Swap all Projector class data members from the source object to another.
         */
        void swap(Projector& that  /**<[in] the source Projector */);

        /**
         * @brief Move assignment operator.
         *
         * Assign a Projector object to another by transferring the data members from the source object.
         *
         * @return the assigned Projector object
         */
        inline Projector& operator=(BOOST_RV_REF(Projector) that  /**<[in] the source Projector */)
        {
            if (this != &that) swap(that);
            return *this;
        }

        /**
         * @brief Check if there is an image to be projected.
         * 
         * @return If there is an image to be projected, return false, otherwise return true.
         */
        bool isEmpty2D() const;

        /**
         * @brief Check if there is a volume to be projected.
         * 
         * @return If there is a volume to be projected, return false, otherwise return true.
         */
        bool isEmpty3D() const;

        /**
         * @brief Return the mode.
         *
         * @return the mode
         */
        int mode() const;

        /**
         * @brief Set the mode.
         */
        void setMode(const int mode  /**<[in] the mode to be set */);

        /**
         * @brief Return the max radius for processing signal in Fourier transform (in pixel).
         *
         * @return the max radius
         */
        int maxRadius() const;

        /**
         * @brief Set the max radius for processing signal in Fourier transform (in pixel).
         */
        void setMaxRadius(const int maxRadius  /**<[in] the max radius to be set */);

        /**
         * @brief Return the interpolation type for this projection.
         *
         * @return the interpolation type
         */
        int interp() const;

        /**
         * @brief Set the interpolation type for this projection.
         */
        void setInterp(const int interp  /**<[in] the interpolation type to be set */);

        /**
         * @brief Return the padding factor.
         *
         * @return the padding factor
         */
        int pf() const;

        /**
         * @brief Set the padding factor.
         */
        void setPf(const int pf  /**<[in] the padding factor to be set*/);

        /**
         * @brief Return a constant reference to the 2D projectee, i.e. the image to be projected.
         *
         * @return a constant reference to the 2D projectee
         */
        const Image& projectee2D() const;

        /**
         * @brief Return a constant reference to the 3D projectee, i.e. the volume to be projected.
         *
         * @return a constant reference to the 3D projectee
         */
        const Volume& projectee3D() const;

        /**
         * @brief Set the source image as the 2D projectee to be projected.
         *
         * Moreover, it automatically sets the max radius of processing signal.
         */
        void setProjectee(Image src  /**<[in] the source image to be set */);

        /**
         * @brief Set the source volume as the 3D projectee to be projected.
         *
         * Moreover, it automatically sets the max radius of processing signal.
         */
        void setProjectee(Volume src  /**<[in] the source volume to be set */);

        /**
         * @brief Project an image, given the rotation matrix.
         */
        void project(Image& dst,        /**<[out] the projected image */
                     const dmat22& mat  /**<[in]  the 2D rotation matrix */
                     ) const;

        /**
         * @brief Project a volume, given the rotation matrix.
         */
        void project(Image& dst,        /**<[out] the projected image */
                     const dmat33& mat  /**<[in]  the 3D rotation matrix */
                     ) const;

        /**
         * @brief Project an image, given the rotation matrix and the pre-determined pixel indices.
         */
        void project(Image& dst,         /**<[out] the projected image */
                     const dmat22& mat,  /**<[in]  the 2D rotation matrix */
                     const int* iCol,    /**<[in]  the index of column */
                     const int* iRow,    /**<[in]  the index of row */
                     const int* iPxl,    /**<[in]  the pixel index */
                     const int nPxl      /**<[in]  the number of pixels */
                     ) const;

        /**
         * @brief Project a volume, given the rotation matrix and the pre-determined pixel indices.
         */
        void project(Image& dst,         /**<[out] the projected image */
                     const dmat33& mat,  /**<[in]  the 3D rotation matrix */
                     const int* iCol,    /**<[in]  the index of column */
                     const int* iRow,    /**<[in]  the index of row */
                     const int* iPxl,    /**<[in]  the pixel index */
                     const int nPxl      /**<[in]  the number of pixels */
                     ) const;

        /**
         * @brief Project an image, given the rotation matrix and the pre-determined pixel indices, while the projected image stored by Complex type.
         */
        void project(Complex* dst,       /**<[out] the projected image, stored by Complex type */
                     const dmat22& mat,  /**<[in]  the 2D rotation matrix */
                     const int* iCol,    /**<[in]  the index of column */
                     const int* iRow,    /**<[in]  the index of row */
                     const int nPxl      /**<[in]  the number of pixels */
                     ) const;

        /**
         * @brief Project a volume, given the rotation matrix and the pre-determined pixel indices, while the projected image stored by Complex type.
         */
        void project(Complex* dst,       /**<[out] the projected image, stored by Complex type */
                     const dmat33& mat,  /**<[in]  the 3D rotation matrix */
                     const int* iCol,    /**<[in]  the index of column */
                     const int* iRow,    /**<[in]  the index of row */
                     const int nPxl      /**<[in]  the number of pixels */
                     ) const;

        /**
         * @brief Project an image using multiple threads, given the rotation matrix.
         */
        void projectMT(Image& dst,        /**<[out] the projected image */
                       const dmat22& mat  /**<[in]  the 2D rotation matrix */
                       ) const;

        /**
         * @brief Project a volume using multiple threads, given the rotation matrix.
         */
        void projectMT(Image& dst,         /**<[out] the projected image */
                       const dmat33& mat,  /**<[in]  the 3D rotation matrix */
                       ) const;

        /**
         * @brief 2D Project using multiple threads, given the rotation matrix and the pre-determined pixel indices.
         */
        void projectMT(Image& dst,         /**<[out] the projected image */
                       const dmat22& mat,  /**<[in]  the 2D rotation matrix */
                       const int* iCol,    /**<[in]  the index of column */
                       const int* iRow,    /**<[in]  the index of row */
                       const int* iPxl,    /**<[in]  the pixel index */
                       const int nPxl      /**<[in]  the number of pixels */
                       ) const;

        /**
         * @brief Project a volume using multiple threads, given the rotation matrix and the pre-determined pixel indices.
         */
        void projectMT(Image& dst,         /**<[out] the projected image */
                       const dmat33& mat,  /**<[in]  the 3D rotation matrix */
                       const int* iCol,    /**<[in]  the index of column */
                       const int* iRow,    /**<[in]  the index of row */
                       const int* iPxl,    /**<[in]  the pixel index */
                       const int nPxl      /**<[in]  the number of pixels */
                       ) const;

        /**
         * @brief Project an image using multiple threads, given the rotation matrix and the pre-determined pixel indices, while the projected image stored by Complex type.
         */
        void projectMT(Complex* dst,       /**<[out] the projected image, stored by Complex type */
                       const dmat22& mat,  /**<[in]  the 2D rotation matrix */
                       const int* iCol,    /**<[in]  the index of column */
                       const int* iRow,    /**<[in]  the index of row */
                       const int nPxl      /**<[in]  the number of pixels */
                       ) const;

        /**
         * @brief Project a volume using multiple threads, given the rotation matrix and the pre-determined pixel indices, while the projected image stored by Complex type.
         */
        void projectMT(Complex* dst,       /**<[out] the projected image, stored by Complex type */
                       const dmat33& mat,  /**<[in]  the 3D rotation matrix */
                       const int* iCol,    /**<[in]  the index of column */
                       const int* iRow,    /**<[in]  the index of row */
                       const int nPxl      /**<[in]  the number of pixels */
                       ) const;

        /**
         * @brief Project an image, given the rotation matrix and the translation vector.
         */
        void project(Image& dst,         /**<[out] the projected image */
                     const dmat22& rot,  /**<[in]  the 2D rotation matrix */
                     const dvec2& t      /**<[in]  the translation vector */
                     ) const;

        /**
         * @brief Project a volume, given the rotation matrix and the translation vector.
         */
        void project(Image& dst,         /**<[out] the projected image */
                     const dmat33& mat,  /**<[in]  the 3D rotation matrix */
                     const dvec2& t      /**<[in]  the translation vector */
                     ) const;

        /**
         * @brief Project an image, given the rotation matrix, the pre-determined pixel indices and the translation vector.
         */
        void project(Image& dst,         /**<[out] the projected image */
                     const dmat22& rot,  /**<[in]  the 2D rotation matrix */
                     const dvec2& t,     /**<[in]  the translation vector */
                     const int* iCol,    /**<[in]  the index of column */
                     const int* iRow,    /**<[in]  the index of row */
                     const int* iPxl,    /**<[in]  the pixel index */
                     const int nPxl      /**<[in]  the number of pixels */
                     ) const;

        /**
         * @brief Project a volume, given the rotation matrix, the pre-determined pixel indices and the translation vector.
         */
        void project(Image& dst,         /**<[out] the projected image */
                     const dmat33& rot,  /**<[in]  the 3D rotation matrix */
                     const dvec2& t      /**<[in]  the translation vector */
                     const int* iCol,    /**<[in]  the index of column */
                     const int* iRow,    /**<[in]  the index of row */
                     const int* iPxl,    /**<[in]  the pixel index*/
                     const int nPxl      /**<[in]  the number of pixels */
                     ) const;

        /**
         * @brief Project an image, given the rotation matrix, the pre-determined pixel indices and the translation vector, while the projected image stored by Complex type.
         */
        void project(Complex* dst,       /**<[out] the projected image, stored by Complex type */
                     const dmat22& rot,  /**<[in]  the 2D rotation matrix */
                     const dvec2& t      /**<[in]  the translation vector */
                     const int nCol,     /**<[in]  the number of columns */
                     const int nRow,     /**<[in]  the number of rows */
                     const int* iCol,    /**<[in]  the index of column */
                     const int* iRow,    /**<[in]  the index of row */
                     const int nPxl      /**<[in]  the number of pixels */
                     ) const;

        /**
         * @brief Project a volume, given the rotation matrix, the pre-determined pixel indices and the translation vector, while the projected image stored by Complex type.
         */
        void project(Complex* dst,       /**<[out] the projected image, stored by Complex type */
                     const dmat33& rot,  /**<[in]  the 3D rotation matrix */
                     const dvec2& t,     /**<[in]  the translation vector */
                     const int nCol,     /**<[in]  the number of columns */
                     const int nRow,     /**<[in]  the number of rows */
                     const int* iCol,    /**<[in]  the index of column */
                     const int* iRow,    /**<[in]  the index of row */
                     const int nPxl      /**<[in]  the number of pixels */
                     ) const;

        /**
         * @brief Project an image using multiple threads, given the rotation matrix and the translation vector.
         */
        void projectMT(Image& dst,         /**<[out] the projected image */
                       const dmat22& rot,  /**<[in]  the 2D rotation matrix */
                       const dvec2& t      /**<[in]  the translation vector */
                       ) const;

        /**
         * @brief Project a volume using multiple threads, given the rotation matrix and the translation vector.
         */
        void projectMT(Image& dst,         /**<[out] the projected image */
                       const dmat33& rot,  /**<[in]  the 3D rotation matrix */
                       const dvec2& t      /**<[in]  the translation vector */
                       ) const;

        /**
         * @brief Project an image using multiple threads, given the rotation matrix, the pre-determined pixel indices and the translation vector.
         */
        void projectMT(Image& dst,         /**<[out] the projected image */
                       const dmat22& rot,  /**<[in]  the 2D rotation matrix */
                       const dvec2& t,     /**<[in]  the translation vector */
                       const int* iCol,    /**<[in]  the index of column */
                       const int* iRow,    /**<[in]  the index of row */
                       const int* iPxl,    /**<[in]  the pixel index */
                       const int nPxl      /**<[in]  the number of pixels */
                       ) const;

        /**
         * @brief Project a volume using multiple threads, given the rotation matrix, the pre-determined pixel indices and the translation vector.
         */
        void projectMT(Image& dst,         /**<[out] the projected image */
                       const dmat33& rot,  /**<[in]  the 3D rotation matrix */
                       const dvec2& t,     /**<[in]  the translation vector */
                       const int* iCol,    /**<[in]  the index of column */
                       const int* iRow,    /**<[in]  the index of row */
                       const int* iPxl,    /**<[in]  the pixel index */
                       const int nPxl      /**<[in]  the number of pixels */
                       ) const;

        /**
         * @brief Project an image using multiple threads, given the rotation matrix, the pre-determined pixel indices and the translation vector, while the projected image stored by Complex type.
         */
        void projectMT(Complex* dst,       /**<[out] the projected image, stored by Complex type */
                       const dmat22& rot,  /**<[in]  the 2D rotation matrix */
                       const dvec2& t,     /**<[in]  the translation vector */
                       const int nCol,     /**<[in]  the number of columns */
                       const int nRow,     /**<[in]  the number of rows */
                       const int* iCol,    /**<[in]  the index of column */
                       const int* iRow,    /**<[in]  the index of row */
                       const int nPxl      /**<[in]  the number of pixels */
                       ) const;

        /**
         * @brief Project a volume using multiple threads, given the rotation matrix, the pre-determined pixel indices and the translation vector, while the projected image stored by Complex type.
         */
        void projectMT(Complex* dst,       /**<[out] the projected image, stored by Complex type */
                       const dmat33& rot,  /**<[in]  the 3D rotation matrix */  
                       const dvec2& t,     /**<[in]  the translation vector */
                       const int nCol,     /**<[in]  the number of columns */
                       const int nRow,     /**<[in]  the number of rows */
                       const int* iCol,    /**<[in]  the index of column */
                       const int* iRow,    /**<[in]  the index of row */
                       const int nPxl      /**<[in]  the number of pixels */
                       ) const;

    private:

        /**
         * @brief Perform gridding correction on projectee.
         */
        void gridCorrection();
};

#endif // PROJECTOR_H
