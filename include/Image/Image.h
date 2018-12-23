/** @file
 *  @author Mingxu Hu
 *  @author Hongkun Yu
 *  @author Liang Qiao
 *  @version 1.4.11.181001
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR     | TIME       | VERSION       | DESCRIPTION
 *  ------     | ----       | -------       | -----------
 *  Liang Qiao | 2018/09/30 | 1.4.11.180930 | add document
 *  Liang Qiao | 2018/10/01 | 1.4.11.181001 | revise the macro brief
 *
 *  @brief Image.h Define the Image class and functions for image processing.
 *
 */

#ifndef IMAGE_H
#define IMAGE_H

#include <cmath>
#include <cstdlib>

#include "Config.h"
#include "Macro.h"
#include "Typedef.h"
#include "Complex.h"
#include "Logging.h"
#include "Precision.h"

#include "Interpolation.h"

#include "ImageBase.h"
#include "BMP.h"

/**
 * @brief This macro loops over each pixel of an image (@f$j@f$ represents the index of row, @f$i@f$ represents the index of column) in real space.
 *
 * @param that the image
 */
#define IMAGE_FOR_EACH_PIXEL_RL(that) \
    for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
        for (int i = -that.nColRL() / 2; i < that.nColRL() / 2; i++)

/**
 * @brief This macro loops over each pixel of an image (@f$j@f$ represents the index of row, @f$i@f$ represents the index of column) in positive half range Fourier space.
 *
 * @param that the image
 */
#define IMAGE_FOR_EACH_PIXEL_FT(that) \
    for (int j = -that.nRowRL() / 2; j < that.nRowRL() / 2; j++) \
        for (int i = 0; i <= that.nColRL() / 2; i++)

/**
 * @brief This macro loops over the pixels of an image in a certain radius (@f$j@f$ represents the index of row, @f$i@f$ represents the index of column) in real space.
 *
 * @param r the radius
 */
#define IMAGE_FOR_PIXEL_R_RL(r) \
    for (int j = -r; j < r; j++) \
        for (int i = -r; i < r; i++)

/**
 * @brief This macro loops over the pixels of an image in a certain radius (@f$j@f$ represents the index of row, @f$i@f$ represents the index of column) in positive half range Fourier space.
 *
 * @param r the radius
 */
#define IMAGE_FOR_PIXEL_R_FT(r) \
    for (int j = -r; j < r; j++) \
        for (int i = 0; i <= r; i++)

/**
 * @brief Compute the indicator of whether the regular pixel is in the conjugate part of volume or not.
 *
 * @return indicator of whether the regular pixel is in the conjugate part of volume or not
 */
inline bool conjHalf(int& iCol,    /**< [in] index of the column (regular voxel) */
                     int& iRow     /**< [in] index of the row (regular voxel) */
                    )
{
    if (iCol >= 0) return false;

    iCol *= -1;
    iRow *= -1;

    return true;
};

/**
 * @brief Compute the indicator of whether the irregular pixel is in the conjugate part of volume or not.
 *
 * @return indicator of whether the irregular pixel is in the conjugate part of volume or not.
 */
inline bool conjHalf(RFLOAT& iCol,   /**< [in] index of the column (irregular voxel) */
                     RFLOAT& iRow    /**< [in] index of the column (irregular voxel) */
                    )
{
    if (iCol >= 0) return false;

    iCol *= -1;
    iRow *= -1;

    return true;
}

/**
 * @brief The Image class stores the content of an image (capable of both real space and Fourier space), and the meta information of this image.
 *
 * This Image class stores the content of an image in an integrated memory array, and the meta information of this image, such as the number of columns and rows. Moreover, this class provides functions for changing and accessing the contents and the meta information.
 */
class Image : public ImageBase
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE(Image)

    protected:

        /**
         * @brief number of columns of the image
         */
        int _nCol;

        /**
         * @brief number of rows of the the image
         */
        int _nRow;
	
        /**
         * @brief number of columns of the image in Fourier space
         */ 
        int _nColFT;

        /**
         * @brief the distances between the irregular voxel and four adjacent regular voxel, helpful for the interpolation in addFT operation
         */ 
        size_t _box[2][2];

    public:

        /**
         * @brief default constructor of Image class.
         */
        Image();

        /**
         * @brief constructor of Image class 
         * 
         * It constructs an Image object with the given number of columns and number of rows in the certain space.
         */
        Image(const int nCol,    /**< [in] number of columns of image*/
              const int nRow,    /**< [in] number of rows of image*/ 
              const int space    /**< [in] image mode: RL_SPACE(real space) or FT_SPACE(Fourier space) */
             );

        /**
         * @brief constructor of Image class
         *
         * It constructs an Image object with an existing Image object, and use the ImageBase, number of columns, rows and columns in Fourier space to initialize this Image object.
         */
        Image(BOOST_RV_REF(Image) that /**< [in] reference of that Image object */) : ImageBase(BOOST_MOVE_BASE(ImageBase, that)),
                                          _nCol(that._nCol),
                                          _nRow(that._nRow),
                                          _nColFT(that._nColFT)
        {
            // _nColFT = that._nColFT;

            FOR_CELL_DIM_2
                _box[j][i] = that._box[j][i];

            that._nCol = 0;
            that._nRow = 0;
            that._nColFT = 0;
        }

        /**
         * @brief default destructor of Image class
         */
        ~Image();

       /**
        * @brief Overload the operator "=", used to swap the contents of Image objects on both sides.
        *
        * @return reference of the Image object 
        */
        inline Image& operator=(BOOST_RV_REF(Image) that /**< [in] reference of that Image object */)
        {
            if (this != &that) swap(that);
            return *this;
        }

        /**
         * @brief Swap that Image object and this Image object.
         */  
        void swap(Image& that /**< [in] reference of that Image object */);

        /**
         * @brief Copy that Image object.
         *
         * @return the copied Image object
         */  
        Image copyImage() const;

        /**
         * @brief This function allocates memory space in a certain space.
         */
        void alloc(const int space /**< [in] image mode: RL_SPACE(real space) or FT_SPACE(Fourier space) */);

        /**
         * @brief This function allocates memory space in a certain space with given number of columns and number of rows.
         */
        void alloc(const int nCol,    /**< [in] number of columns of image */
                   const int nRow,    /**< [in] number of rows of image */
                   const int space    /**< [in] image mode: RL_SPACE(real space) or FT_SPACE(Fourier space) */
                  );

        /**
         * @brief This function returns the number of columns of this image in real space.
         *
         * @return number of columns of this image in real space
         */
        inline int nColRL() const { return _nCol; };

        /**
         * @brief This function returns the number of rows of this image in real space.
         *
         * @return number of rows of this image in real space
         */
        inline int nRowRL() const { return _nRow; };

        /**
         * @brief This function returns the number of columns of this image in Fourier space.
         *
         * @return number of columns of this image in Fourier space
         */
        inline int nColFT() const { return _nColFT; };

        /**
         * @brief This function returns the number of rows of this image in Fourier space.
         *
         * @return number of rows of this image in Fourier space
         */
        inline int nRowFT() const { return _nRow; };

        /**
         * @brief This function saves the real space image to a BMP image. 
         *
         * If the file does not exist, create it.
         */
        void saveRLToBMP(const char* filename /**< [in] the file name of the BMP image */) const;

        /**
         * @brief This function saves the Fourier space image to a BMP image. 
         *
         * If the file does not exist, create it.
         */
        void saveFTToBMP(const char* filename,     /**< [in] the file name of the BMP image */
                         const RFLOAT c            /**< [in] factor for caculating the value in BMP image, @f$\log (1+mod(x)\times c)@f$ */
                        ) const;

        /**
         * @brief This function gets the value of the pixel at the certain column and row in real space.
         *
         * @return value of the pixel
         */
        RFLOAT getRL(const int iCol,    /**< [in] index of the column (regular voxel) of this image in real space */
                     const int iRow     /**< [in] index of the row (regular voxel) of this image in real space */
                    ) const;

        /**
         * @brief This function sets the value of the pixel at the certain column and row in real space.
         */
        void setRL(const RFLOAT value,    /**< [in] value of the pixel to set */
                   const int iCol,        /**< [in] index of the column (regular voxel) of this image in real space */
                   const int iRow         /**< [in] index of the row (regular voxel) of this image in real space */ 
                  );

        /**
         * @brief This function returns the value of the pixel at the certain column and row in Fourier space.
         *
         * @return complex value of the pixel at certain column and row in Fourier space
         */
        Complex getFT(int iCol,    /**< [in] index of the column (regular voxel) of this image in Fourier space */
                      int iRow     /**< [in] index of the rows (regular voxel) of this image in Fourier space */
                     ) const;

        /**
         * @brief This function returns the value of the pixel at the certain column and row in positive half range Fourier space.
         *
         * @return complex value of the pixel at certain column and row in positive half range Fourier space
         */
        Complex getFTHalf(const int iCol,  /**< [in] index of the column (regular voxel) of this image in positive half range Fourier space */
                          const int iRow   /**< [in] index of the column (regular voxel) of this image in positive half range Fourier space */
                         ) const;

        /**
         * @brief This function sets the complex value of the pixel at the certain column and row in Fourier space.
         */
        void setFT(const Complex value,    /**< [in] value of the pixel to set in Fourier space */
                   int iCol,               /**< [in] index of the column (regular voxel) of this image in Fourier space */
                   int iRow                /**< [in] index of the row (regular voxel) of this image in Fourier space */ 
                  );

        /**
         * @brief This function sets the complex value of the pixel at the certain column and row in positive half range Fourier space.
         */
        void setFTHalf(const Complex value,    /**< [in] value of the pixel to set in positive half range Fourier space */
                       int iCol,               /**< [in] index of the column (regular voxel) of this image in positive half range Fourier space */
                       int iRow                /**< [in] index of the row (regular voxel) of this image in positive half range Fourier space */ 
                      );

        /**
         * @brief This function adds the complex value of the pixel at the certain column and row in Fourier space.
         */
        void addFT(const Complex value,   /**< [in] the complex value to add in Fourier space */
                   const int iCol,        /**< [in] index of the column (regular voxel) of this image in Fourier space */
                   const int iRow         /**< [in] index of the row (regular voxel) of this image in Fourier space */
                  );

        /**
         * @brief This function adds the complex value of the pixel at the certain column and row in positive half range Fourier space.
         */
        void addFTHalf(const Complex value,   /**< [in] the complex value to add in positive half range Fourier space */
                       const int iCol,        /**< [in] index of the column (regular voxel) of this image in positive half range Fourier space */
                       const int iRow         /**< [in] index of the row (regular voxel) of this image in positive half range Fourier space */
                      );

        /**
         * @brief This function adds the real-part value of the pixel at the certain column and row in Fourier space.
         */
        void addFT(const RFLOAT value,    /**< [in] the real-part value to add in Fourier space */
                   const int iCol,        /**< [in] index of the column (regular voxel) of this image in Fourier space */
                   const int iRow         /**< [in] index of the row (regular voxel) of this image in Fourier space */
                  );

        /**
         * @brief This function adds the real-part value of the pixel at the certain column and row in positive half range Fourier space.
         */
        void addFTHalf(const RFLOAT value,    /**< [in] the real-part value to add in positive half range Fourier space */
                       const int iCol,        /**< [in] index of the column (regular voxel) of this image in positive half range Fourier space */
                       const int iRow         /**< [in] index of the row (regular voxel) of this image in positive half range Fourier space */
                      );

        /*
        RFLOAT getBiLinearRL(const RFLOAT iCol,   
                             const RFLOAT iRow     
			    ) const;

        Complex getBiLinearFT(const RFLOAT iCol,    
                              const RFLOAT iRow     
			     ) const;
	*/
        
        /**
         * @brief This function gets the complex value of the  irregular voxel in Fourier space by the specific interpolation method.
         *
         * @return the complex value of an irregular voxel in Fourier space spce by interpolation methods
         */
        Complex getByInterpolationFT(RFLOAT iCol,       /**< [in] index of the column (irregular voxel) of this image in Fourier space */
                                     RFLOAT iRow,       /**< [in] index of thr row (irregular voxel) of this image in Fourier space */
                                     const int interp   /**< [in] type of interpolation methods, (NEAREST_INTERP or LINEAR_INTERP) */
                                    ) const;

        /**
         * @brief This function adds the complex value to the certain irregular voxel in Fourier space.
         */
        void addFT(const Complex value,   /**< [in] the complex value to add */
                   RFLOAT iCol,           /**< [in] index of the column (irregular voxel) of this image in Fourier space */
                   RFLOAT iRow            /**< [in] index of the row (irregular voxel) of this image in Fourier space */
                  );

        /**
         * @brief This function adds the real-part  value to the certain irregular voxel in Fourier space.
         */
        void addFT(const RFLOAT value,    /**< [in] the real-part value to add */
                   RFLOAT iCol,           /**< [in] index of the column (irregular voxel) of this image in Fourier space */
                   RFLOAT iRow            /**< [in] index of the row (irregular voxel) of this image in Fourier space */
                  );

        /**
         * @brief Clear the Image object.
         */ 
        void clear();

        /**
         * @brief  Compute the index of the regular voxel in real space.
         *
         * @return index of the regular voxel
         */ 
        inline int iRL(const int i,    /**< [in] column index  of the regular voxel of this image */
                       const int j     /**< [in] row index of the regular voxel of this image */
                      ) const
        {
            return (j >= 0 ? j : j + _nRow) * _nCol
                 + (i >= 0 ? i : i + _nCol);
        }

        /**
         * @brief Compute the index of the regular voxel in Fourier space.
         *
         * @return index of the regular voxel
         */
        inline int iFT(int i,    /**< [in] column index of the regular voxel of this image */
                       int j     /**< [in] row index of the regular voxel of this image */
                      ) const
        {
            if (i >= 0)
                return iFTHalf(i, j);
            else
                return iFTHalf(-i, -j);
        }

	/**
	 * @brief Compute the index of the regular voxel in Fourier space and judge the conjugate area.
	 *
	 * @return index of the regular voxel
	 */
        inline int iFT(bool& conj,     /**< [out] indicator of whether the regular voxel locates in the conjugate part or not */
                       int i,          /**< [in] column index of the regular voxel of this image */
                       int j           /**< [in] row index of the regular voxel of this image */
		      ) const
        {
            conj = conjHalf(i, j);

            return iFTHalf(i, j);
        }

        /**
         * @brief Compute the index of the regular voxel in positive half range Fourier space..
         *
         * @return index of the regular voxel
         */
        inline int iFTHalf(const int i,    /**< [in] column index of the regular voxel of this image*/
                           const int j     /**< [in] row index of the regular voxel of this image */
                          ) const
        {
            return (j >= 0 ? j : j + _nRow) * _nColFT + i;
            //return 0;
        }

    private:
        /**
         *  @brief Initialize the box matrix.
         */
        void initBox();

        /**
         * @brief This function checks whether the given coordinates is in the boundary of the image or not in real space. 
         *
         * If not, it will crash the process and record a fatal log.
         */
        void coordinatesInBoundaryRL(const int iCol,    /**< [in] index of the column(grid point) of the image in real space */
                                     const int iRow     /**< [in] index of the row(grid point) of the image in real space */
                                    ) const;

        /**
         * @brief This function checks whether the given coordinates is in the boundary of the image or not in Fourier space. 
         *
         * If not, it will crash the process and record a fatal log.
         */
        void coordinatesInBoundaryFT(const int iCol,    /**< [in] index of the column(grid point) of the image in Fourier space */
                                     const int iRow     /**< [in] index of the row(grid point) of the image in Fourier space */
                                    ) const;

        /**
         * @brief This function gets the complex value of regular voxel in positive half range Fourier space.
         *
         * @return complex value of the regular voxel in positive half range Fourier space
         */
        Complex getFTHalf(const RFLOAT w[2][2],    /**< [in] weights of adjacent four voxels */
                          const int x0[2]          /**< [in] index of the core voxel in positive half range Fourier space */
                         ) const;

        /**
         * @brief Add a certain complex value on the voxel in Fourier space at given coordinates.
         */
        void addFTHalf(const Complex value,     /**< [in] complex value to be added */
                       const RFLOAT w[2][2],    /**< [in] weights for value to add */
                       const int x0[2]          /**< [in] index of the core voxel in positive half range Fourier space */
                      );

        /**
         * @brief Add the real-part value to the certain regular voxel in positive half range Fourier space.
         */
        void addFTHalf(const RFLOAT value,     /**< [in] real-part value to be added */
                       const RFLOAT w[2][2],   /**< [in] weights for value to add */
                       const int x0[2]         /**< [in] index of the regular voxel in positive half range Fourier space */
                      );
};

#endif // IMAGE_H
