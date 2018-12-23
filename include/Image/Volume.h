/** @file
 *  @author Mingxu Hu
 *  @author He Zhao
 *  @version 1.4.11.180913
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR    | TIME       | VERSION       | DESCRIPTION
 *  ------    | ----       | -------       | -----------
 *  Mingxu Hu | 2015/03/23 | 0.0.1.050323  | new file
 *  He Zhao   | 2018/09/13 | 1.4.11.180913 | add notes & header
 *
 *  @brief Volume.h contains several macros that loop over each voxel in real or Fourier space, definitions of class members, functions of allocating or freeing space, returning index, getting voxel value, adding voxel value, clearing up space, reseting the volume. 
 */

  
#ifndef VOLUME_H
#define VOLUME_H

#include <cmath>
#include <cstdlib>

#include "omp_compat.h"

#include "Config.h"
#include "Macro.h"
#include "Typedef.h"
#include "Precision.h"
#include "Complex.h"
#include "Logging.h"

#include "ImageBase.h"
#include "BMP.h"
#include "Image.h"
#include "Interpolation.h"
#include "Functions.h"
#include "TabFunction.h"
#include "Coordinate5D.h"

/**
 * @brief This macro loops over each pixel within a sphere of which origin is a voxel in Fourier space.
 */
#define VOLUME_SUB_SPHERE_FT(a /**< [in] the radius of this sphere */ \
                            ) \
    for (int k = GSL_MAX_INT(-_nSlc / 2, FLOOR(iSlc - a)); \
             k <= GSL_MIN_INT(_nSlc / 2 - 1, CEIL(iSlc + a)); \
             k++) \
        for (int j = GSL_MAX_INT(-_nRow / 2, FLOOR(iRow - a)); \
                 j <= GSL_MIN_INT(_nRow / 2 - 1, CEIL(iRow + a)); \
                 j++) \
            for (int i = GSL_MAX_INT(-_nCol / 2, FLOOR(iCol - a)); \
                     i <= GSL_MIN_INT(_nCol / 2, CEIL(iCol + a)); \
                     i++)

/**
 * @brief This macro loops over each pixel within a sphere of which origin is a voxel in real space.
 */
#define VOLUME_SUB_SPHERE_RL(a /**< [in] the radius of this sphere */ \
                            ) \
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
 * @brief This macro loops over each pixel of a volume in real space.
 *
 * @param that the volume
 */
#define VOLUME_FOR_EACH_PIXEL_RL(that) \
    for (int k = -that.nSlcRL() / 2; k < that.nSlcRL() / 2; k++) \
        for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
            for (int i = -that.nColRL() / 2; i < that.nColRL() / 2; i++) \

/**
 * @brief This macro loops over each pixel of a volume in Fourier space.
 *
 * @param that the volume
 */
#define VOLUME_FOR_EACH_PIXEL_FT(that) \
    for (int k = -that.nSlcRL() / 2; k < that.nSlcRL() / 2; k++) \
        for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
            for (int i = 0; i <= that.nColRL() / 2; i++)

/**
 * @brief This macro loops over each pixel of a volume in a certain radius in real space.
 *
 * @param r the radius
 */
#define VOLUME_FOR_PIXEL_R_RL(r) \
    for (int k = -r; k < r; k++) \
        for (int j = -r; j < r; j++) \
            for (int i = -r; i < r; i++)

/**
 * @brief This macro loops over each pixel of a volume in a certain radius in Fourier space.
 *
 * @param r the radius
 */
#define VOLUME_FOR_PIXEL_R_FT(r) \
    for (int k = -r; k < r; k++) \
        for (int j = -r; j < r; j++) \
            for (int i = 0; i <= r; i++)

/**
 * @brief Compute the indicator of whether the regular pixel is in the conjugate part of volume or not.
 *
 * @return indicator of whether the regular pixel is in the conjugate part of volume or not.
 */
inline bool conjHalf(int& iCol, /**< [in] column index of the regular voxel in real space */
                     int& iRow, /**< [in] row index of the regular voxel in real space */
                     int& iSlc  /**< [in] slice index of the regular voxel in real space */
                    )
{
    if (iCol >= 0) return false;

    iCol *= -1;
    iRow *= -1;
    iSlc *= -1;

    return true;
};

/**
 * @brief Compute the indicator of whether the irregular pixel is in the conjugate part of volume or not.
 *
 * @return indicator of whether the irregular pixel is in the conjugate part of volume or not.
 */
inline bool conjHalf(RFLOAT& iCol, /**< [in] column index of the irregular voxel in real space */
                     RFLOAT& iRow, /**< [in] row index of the irregular voxel in real space */
                     RFLOAT& iSlc  /**< [in] slice index of the irregular voxel in real space */
                    )
{
    if (iCol >= 0) return false;

    iCol *= -1;
    iRow *= -1;
    iSlc *= -1;

    return true;
}

/**
 * @brief The Volume class stores the content of a volume (capable of both real space and Fourier space), and the meta information of this image.
 *
 * This Volume class stores the content of a volume in an integrated memory array, and the meta information of this volume, such as the number of columns, rows and slices. Moreover, this class provides functions for changing and accessing the contents and the meta information.
 */
class Volume : public ImageBase
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE(Volume)

    public:

        /**
         * number of columns of the volume
         */
        int _nCol;

        /**
         * number of rows of the volume
         */
        int _nRow;

        /**
         * number of slices of the volume
         */
        int _nSlc;

        /**
         * number of columns of the volume in Fourier space
         */
        int _nColFT;

        /**
         * the distances between the irregular(non-grid) voxel and eight adjacent regular voxels, helpful for the interpolation in addFT operation
         */
        size_t _box[2][2][2];

    public:

        /**
         * @brief Create default volume.
         */
        Volume();

        /**
         * @brief Create designated-size volume.
         */
        Volume(const int nCol, /**< [in] number of columns of this volume */
               const int nRow, /**< [in] number of rows of this volume */
               const int nSlc, /**< [in] number of slices of this volume */
               const int space /**< [in] the space this volume allocating in, where RL_SPACE stands for the real space and FT_SPACE stands for the Fourier space */
              );

        /**
         * @brief Move volume. Exchange the volume pointed by this and that volume pointed by that.
         */
        Volume(BOOST_RV_REF(Volume) that /**< [in] the original volume */
              ) : ImageBase(BOOST_MOVE_BASE(ImageBase, that)),
                                            _nCol(that._nCol),
                                            _nRow(that._nRow),
                                            _nSlc(that._nSlc),
                                            _nColFT(that._nColFT)
        {
            // _nColFT = that._nColFT;

            FOR_CELL_DIM_3
                _box[k][j][i] = that._box[k][j][i];

            that._nCol = 0;
            that._nRow = 0;
            that._nSlc = 0;

            that._nColFT = 0;
        }

        /**
         * @brief Default deconstructor. Automatically free all allocated space.
         */
        ~Volume();

        /**
         * @brief Exchange the instance pointed by this and the instance pointed by that.
         */
        inline Volume& operator=(BOOST_RV_REF(Volume) that /**< [in] the original volume */)
        {
            if (this != &that) swap(that);
            return *this;
        }

        /**
         * @brief Swap volume. Exchange the instance pointed by this and the instance pointed by that.
         */
        void swap(Volume& that);

        /**
         * @brief Copy volume.
         *
         * @return the original volume.
         */
        Volume copyVolume() const;

        /**
         * @brief Allocate a block of memory in designated space.
         */
        void alloc(const int space /**< [in] the space this volume allocating, where RL_SPACE stands for the real space and FT_SPACE stands for the Fourier space */
                  );

        /**
	 * @brief Allocate a block of memory in designated space.
         */
        void alloc(const int nCol, /**< [in] number of columns of this volume */
                   const int nRow, /**< [in] number of rows of this volume */
                   const int nSlc, /**< [in] number of slices of this volume */
                   const int space /**< [in] the space this volume allocating, where RL_SPACE stands for the real space and FT_SPACE stands for the Fourier space */
                  );

        /**
         * @brief Return the number of columns of this volume in real space.
         *
         * @return the number of columns of this volume in real space.
         */
        inline int nColRL() const 
        { 
            return _nCol; 
        };

        /**
         * @brief Return the number of rows of this volume in real space.
         *
         * @return the number of rows of this volume in real space.
         */
        inline int nRowRL() const 
        { 
            return _nRow; 
        };

        /**
         * @brief Return the number of rows of this volume in real space.
         *
         * @return the number of rows of this volume in real space.
         */
        inline int nSlcRL() const 
        { 
            return _nSlc; 
        };

        /**
         * @brief Return the number of columns of this volume in Fourier space.
         *
         * @return the number of columns of this volume in Fourier space.
         */
        inline int nColFT() const 
        { 
            return _nColFT; 
        };

        /**
         * @brief Return the number of rows of this volume in Fourier space.
         *
         * @return the number of rows of this volume in Fourier space.
         */
        inline int nRowFT() const 
        { 
            return _nRow; 
        };

        /**
         * @brief Return the number of slices of this volume in Fourier space.
         *
         * @return the number of slices of this volume in Fourier space.
         */
        inline int nSlcFT() const 
        { 
            return _nSlc; 
        };

        /**
         * @brief Get the value of the voxel in real space at given coordinates.
         *
         * @return the value of the voxel in real space at given cooridnates.
         */
        RFLOAT getRL(const int iCol, /**< [in] column index of the voxel in real space */
                     const int iRow, /**< [in] row index of the voxel in real space */
                     const int iSlc  /**< [in] slice index of the voxel in real space */
                    ) const;

        /**
         * @brief Set the value of the voxel in real space at given coordinates.
         */
        void setRL(const RFLOAT value, /**< [in] the value of the voxel in real space */
                   const int iCol,     /**< [in] column index of the voxel in real space */
                   const int iRow,     /**< [in] row index of the voxel in real space */
                   const int iSlc      /**< [in] slice index of the voxel in real space */
                  );

        /**
         * @brief Add a certain value on the voxel in real space at given coordinates.
         */
        void addRL(const RFLOAT value, /**< [in] the value of the voxel in real space */
                   const int iCol,     /**< [in] column index of the voxel in real space */
                   const int iRow,     /**< [in] row index of the voxel in real space */
                   const int iSlc      /**< [in] slice index of the voxel in real space */
                  );

        /**
         * @brief Get the value of the voxel in Fourier space at given coordinates.
         *
         * @return the value of the voxel in Fourier space at given cooridnates.
         */
        Complex getFT(int iCol, /**< [in] column index of the voxel in Fourier space */
                      int iRow, /**< [in] row index of the voxel in Fourier space */
                      int iSlc  /**< [in] slice index of the voxel in Fourier space */
                     ) const;

        /**
         * @brief Get the value of the voxel in the positive half of Fourier space at given cooridnates.
         *
         * @return the value of the voxel in the positive half of Fourier space at given cooridnates.
         */
        Complex getFTHalf(const int iCol, /**< [in] column index of the voxel in Fourier space */
                          const int iRow, /**< [in] row index of the voxel in Fourier space */
                          const int iSlc  /**< [in] slice index of the voxel in Fourier space */
                         ) const;

        /**
         * @brief Set the value of the regular voxel in the whole Fourier space at given coordinateS.
         */
        void setFT(const Complex value, /**< [in] the value of regular voxel in Fourier space */
                   int iCol,            /**< [in] column index of the regular voxel in Fourier space */
                   int iRow,            /**< [in] row index of the regular voxel in Fourier space */
                   int iSlc             /**< [in] slice index of the regular voxel in real space */
                  );

        /**
         * @brief Set the value of the regular voxel in the positive half of Fourier space at given coordinates.
         */
        void setFTHalf(const Complex value, /**< [in] the value of the regular voxel in Fourier space */
                       const int iCol,      /**< [in] column index of the regular voxel in Fourier space */
                       const int iRow,      /**< [in] row index of the regular voxel in Fourier space */
                       const int iSlc       /**< [in] slice index of the regular voxel in Fourier space */
                      );
        
        /**
         * @brief Add a certain value on a regular voxel in Fourier space at given coordinate.
         */
        void addFT(const Complex value, /**< [in] the value of the regular voxel */
                   int iCol,            /**< [in] column index of the regular voxel in Fourier space */
                   int iRow,            /**< [in] row index of the regular voxel in Fourier space */
                   int iSlc             /**< [in] slice index of the regular voxel in real space */
                  );

        /**
         * @brief Add a certain value on a regular voxel in Fourier space at given coordiantes.
         */
        void addFTHalf(const Complex value, /**< [in] the value of the regular voxel */
                       const int iCol,      /**< [in] column index of the regular voxel in Fourier space */
                       const int iRow,      /**< [in] row index of the regular voxel in Fourier space */
                       const int iSlc       /**< [in] slice index of the regular voxel in real space */
                      );

        /**
         * @brief Add the real part on a regular voxel in Fourier space at given coordinates.
         */
        void addFT(const RFLOAT value, /**< [in] the real part of the regular voxel */
                   int iCol,           /**< [in] column index of the regular voxel in Fourier space */
                   int iRow,           /**< [in] row index of the regular voxel in Fourier space */
                   int iSlc            /**< [in] slice index of the regular voxel in real space */
                  );

        /**
         * @brief Add the real part on a regular voxel in Fourier space at given coordinates.
         */
        void addFTHalf(const RFLOAT value, /**< [in] the real part of the regular voxel */
                       const int iCol,     /**< [in] column index of the regular voxel in Fourier space */
                       const int iRow,     /**< [in] row index of the regular voxel in Fourier space */
                       const int iSlc      /**< [in] slice index of the regular voxel in real space */
                      );

        /**
         * @brief Return the value of an irregular(non-grid) voxel in real space by interpolation.
         *
         * @return the value of an irregular(non-grid) voxel in real space by interpolation.
         */
        RFLOAT getByInterpolationRL(const RFLOAT iCol, /**< [in] column index of the irregular voxel in real space */
                                    const RFLOAT iRow, /**< [in] row index of the irregular voxel in real space */
                                    const RFLOAT iSlc, /**< [in] slice index of the irregular voxel in real space */
                                    const int interp   /**< [in] indicator of the type of interpolation, where INTERP_NEAREST stands for the nearest point interpolation, INTERP_LINEAR stands for the trilinear interpolation and INTERP_SINC stands for the sinc interpolation */
                                   ) const;

        /**
         * @brief Return the value of an irregular(non-grid) voxel in Fourier space by interpolation.
         *
         * @return the value of an irregular(non-grid) voxel in Fourier spce by interpolation.
         */
        Complex getByInterpolationFT(RFLOAT iCol,     /**< [in] column index of the irregular voxel in Fourier space */
                                     RFLOAT iRow,     /**< [in] row index of the irregular voxel in Fourier space */
                                     RFLOAT iSlc,     /**< [in] slice index of the irregular voxel in Fourier space */
                                     const int interp /**< [in] indicator of the type of interpolation, where INTERP_NEAREST stands for the nearest point interpolation, INTERP_LINEAR stands for the trilinear interpolation and INTERP_SINC stands for the sinc interpolation */
                                    ) const;

        /**
         * @brief Add a certain complex value on the irregular(non-grid) voxel in Fourier space at given coordinates.
         */
        void addFT(const Complex value, /**< [in] value to be added */
                   RFLOAT iCol,         /**< [in] column index of the irregular voxel in real space */
                   RFLOAT iRow,         /**< [in] row index of the irregular voxel in real space */
                   RFLOAT iSlc          /**< [in] slice index of the irregular voxel in real space */
                   );

        /**
         * @brief Add a certain real value on the irregular(non-grid) voxel in Fourier space at given coordinates.
         */
        void addFT(const RFLOAT value, /**< [in] value to be added */
                   RFLOAT iCol,        /**< [in] column index of the irregular voxel in real space */
                   RFLOAT iRow,        /**< [in] row index of the irregular voxel in real space */
                   RFLOAT iSlc         /**< [in] slice index of the irregular voxel in real space */
                  );

        /**
         * @brief Add a certain value on an irregular(non-grid) voxel in Fourier space by a kernal of Modified Kaiser Bessel Function.
         */
        void addFT(const Complex value, /**< [in] value to be added */
                   const RFLOAT iCol,   /**< [in] column index of the irregular voxel in real space */
                   const RFLOAT iRow,   /**< [in] row index of the irregular voxel in real space */
                   const RFLOAT iSlc,   /**< [in] slice index of the irregular voxel in real space */
                   const RFLOAT a,      /**< [in] radius of Modified Kaiser Bessel Function */
                   const RFLOAT alpha   /**< [in] smooth factor of Modified Kaiser Bessel Function */
                  );

        /**
         * @brief Add a certain value on the real part of an irregular(non-grid) voxel in Fourier space by a kernal of Modified Kaiser Bessel Function.
         */
        void addFT(const RFLOAT value, /**< [in] value to be added */
                   const RFLOAT iCol,  /**< [in] column index of the irregular voxel in real space */
                   const RFLOAT iRow,  /**< [in] row index of the irregular voxel in real space */
                   const RFLOAT iSlc,  /**< [in] slice index of the irregular voxel in real space */
                   const RFLOAT a,     /**< [in] radius of Modified Kaiser Bessel Function */
                   const RFLOAT alpha  /**< [in] smooth factor of Modified Kaiser Bessel Function */
                  );

        /**
         * @brief Add a certain value on an irregular(non-grid) voxel in Fourier space by a certain kernel.
         */
        void addFT(const Complex value,      /**< [in] value to be added */
                   const RFLOAT iCol,        /**< [in] column index of the irregular voxel in real space */
                   const RFLOAT iRow,        /**< [in] row index of the irregular voxel in real space */
                   const RFLOAT iSlc,        /**< [in] slice index of the irregular voxel in real space */
                   const RFLOAT a,           /**< [in] radius of the blob */
                   const TabFunction& kernel /**< [in] a tabular function indicating the kernel which is a function of only one parameter, the square of radius */
                  );

        /**
         * @brief Add a certain value on the real part of an irregular(non-grid) voxel in Fourier space by a certain kernel.
         */
        void addFT(const RFLOAT value,       /**< [in] value to be added */
                   const RFLOAT iCol,        /**< [in] column index of the irregular voxel in real space */
                   const RFLOAT iRow,        /**< [in] row index of the irregular voxel in real space */
                   const RFLOAT iSlc,        /**< [in] slice index of the irregular voxel in real space */
                   const RFLOAT a,           /**< [in] radius of the blob */
                   const TabFunction& kernel /**< [in] a tabular function indicating the kernel which is a function of only one paramter, the square of radius */
                  );

        /**
         * @brief Clear up the allocated space and reset volume size to 0.
         */
        void clear();

        /**
         * @brief Compute real-space index of the regular voxel.
         *
         * @return real-space index of the regular voxel.
         */
        inline size_t iRL(const int i, /**< [in] column index of the regular voxel in real space */
                          const int j, /**< [in] row index of the regular voxel in real space */
                          const int k  /**< [in] slice index of the regular voxel in real space */
                         ) const
        {
            return (k >= 0 ? k : k + _nSlc) * _nCol * _nRow
                 + (j >= 0 ? j : j + _nRow) * _nCol
                 + (i >= 0 ? i : i + _nCol);
        }

        /**
         * @brief Compute Fourier-space index of the regular voxel.
         *
         * @return Fourier-space index of the regular voxel.
         */
        inline size_t iFT(int i, /**< [in] column index of the regular voxel in Fourier space */
                          int j, /**< [in] row index of the regular voxel in Fourier space */
                          int k  /**< [in] slice index of the regular voxel in Fourier space */
                         ) const
        {
            if (i >= 0)
                return iFTHalf(i, j, k);
            else
                return iFTHalf(-i, -j, -k);
        }

        /**
         * @brief Compute Fourier-space index of the regular voxel in the whole space, including the positive part and the conjugate part.
         *
         * @return Fourier-space index of the regular voxel in the whole space, including the positive half part and the conjugate part.
         */
        inline size_t iFT(bool& conj, /**< [out] indicator of whether the regular voxel locates in the conjugate part or not */
                          int i,      /**< [in] column index of the regular voxel in Fourier space */
                          int j,      /**< [in] row index of the regular voxel in Fourier space */
                          int k       /**< [in] slice index of the regular voxel in Fourier space */
                         ) const
        {
            conj = conjHalf(i, j, k);

            return iFTHalf(i, j, k);
        }

        /**
         * @brief Compute Fourier-space index of the regular voxel in the positive half space.
         *
         * @return Fourier-space index of the regular voxel in the positive half space.
         */
        inline size_t iFTHalf(const int i, /**< [in] column index of the regular voxel in Fourier space */
                              const int j, /**< [in] row index of the regular voxel in Fourier space */
                              const int k  /**< [in] slice index of the regular voxel in Fourier space */
                              ) const
        {
            return (k >= 0 ? k : k + _nSlc) * _nColFT * _nRow
                 + (j >= 0 ? j : j + _nRow) * _nColFT 
                 + i;
        }

    private:

        /**
         * @brief Initialize the box. 
         */
        void initBox();

        /**
         * @brief Check whether the given coordinates is within the boundary of the volume in real space or not. If not, it will crash the process and record a fatal log.
         */
        void coordinatesInBoundaryRL(const int iCol, /**< [in] column index of the voxel in real space */
                                     const int iRow, /**< [in] row index of the voxel in real space */
                                     const int iSlc  /**< [in] slice index of the voxel in real space */
                                    ) const;

        /**
         * @brief Check whether the given coordinates is within the boundary of the volume in Fourier space or not. If not, it will crash the process and record a fatal log.
         */
        void coordinatesInBoundaryFT(const int iCol, /**< [in] column index of the voxel in Fourier space */
                                     const int iRow, /**< [in] row index of the voxel in Fourier space */
                                     const int iSlc  /**< [in] slice index of the voxel in Fourier space */
                                     ) const;

        /**
         * @brief Get the value of the voxel in real space at given coordinates.
         *
         * @return the value of the voxel in real space at given coordinates.
         */
        RFLOAT getRL(const RFLOAT w[2][2][2], /**< [in] weights of adjacent eight voxels */
                     const int x0[3]          /**< [in] index of the core voxel in real space */
                    ) const;

        /**
         * @brief Get the value of the voxel in the positvie part of Fourier space at given coordinates.
         *
         * @return the value of the voxel in the positive part of Fourier space at given coordinates.
         */
        Complex getFTHalf(const RFLOAT w[2][2][2], /**< [in] weights of adjacent eight voxels */
                          const int x0[3]          /**< [in] index of the core voxel in Fourier space */
                         ) const;

        /**
         * @brief Add a certain complex value on the voxel in Fourier space at given coordinates.
         */
        void addFTHalf(const Complex value,     /**< [in] value to be added */
                       const RFLOAT w[2][2][2], /**< [in] weights of adjacent eight voxels */
                       const int x0[3]          /**< [in] index of the core voxel in Fourier space */
                      );

        /**
         * @brief Add a certain real value on the voxel in Fourier space at given coordinates.
         */
        void addFTHalf(const RFLOAT value,      /**< [in] value to be added */
                       const RFLOAT w[2][2][2], /**< [in] weights of adjacent eight voxels */
                       const int x0[3]          /**< [in] index of the core voxel in Fourier space */
                      );
};

#endif // VOLUME_H
