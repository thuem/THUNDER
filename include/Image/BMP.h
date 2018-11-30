/** @file
 *  @author Mingxu Hu
 *  @author Hongkun Yu
 *  @author Fande Yu
 *  @version 1.4.11.080913
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR    | TIME       | VERSION       | DESCRIPTION
 *  ------    | ----       | -------       | -----------
 *  Mingxu Hu | 2015/03/23 | 0.0.1.050323  | new file
 *  Fande Yu  | 2018/09/17 | 1.4.11.080913 | annotation
 * 
 *
 *  @brief BMP.h contains the BMP file format and related functions.
 *
 *  BMP.h only contains the whole construction of a new BMP image file, excluding the modification of a existed BMP file.
 */


 
#ifndef BMP_H
#define BMP_H


#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>

/**
 * @brief The BITMAPFILEHEADER structure contains information about the type, size, and layout of a file that contains a DIB.
 */
typedef struct 
{
    /** 
     * @brief type of bit map file, the header field used to identify the BMP and DIB file is 0x42 0x4D in hexadecimal, same as BM in ASCII. The following entries are possible:
     *  *BM – Windows 3.1x, 95, NT, ... etc.
     *  *BA – OS/2 struct bitmap array
     *  *CI – OS/2 struct color icon
     *  *CP – OS/2 const color pointer
     *  *IC – OS/2 struct icon
     *  *PT – OS/2 pointer 
     */
    unsigned short bfType;    
    
    unsigned bfSize;  /**< size of the BMP file in bytes */

    /**
     * @brief Reserved for futural use;
     *
     * actual value depends on the application that creates the image
     */
    unsigned short bfReserved1;  

    /**
     * @brief Reserved for futural use;
     *
     * actual value depends on the application that creates the image 
     */
    unsigned short bfReserved2;  

    /**
     * @brief offset,
     *
     * i.e starting address of the byte where the bitmap image data (pixel array) can be found 
     */
    unsigned bfoffBits;          

} __attribute__ ((packed)) BITMAPFILEHEADER; 

/**
 * @brief Bitmap information header
 *
 * This block of bytes tells the application detailed information about the image, which will be used to display the image on the screen. 
 */
typedef struct 
{ 
    unsigned biSize;            /**< the size of this header */
    unsigned biWidth;           /**< the width of bit map, measured in pixel */
    unsigned biHeight;          /**< the height of bit map, measured in pixel */
    unsigned short biPlanes;    /**< the number if color planes, must be 1 */

    /**
     * @brief  the number of bits per pixel,which is the color depth of the inmage.
     *
     * Typical values are 1, 4, 8, 16, 24, and 32 
     */
    unsigned short biBitCount;  
    unsigned biCompress;        /**< the compression method being used. */ 
    unsigned biSizeImage;       /**< the image size */ 
    unsigned biXPelsPerMeter;   /**< the horizontal resolution of the image.(pixel per meter) */
    unsigned biYPelsPerMeter;   /**< the vertical reselution of the image.(pixel per meter)*/

    /**
     * @brief  the number of color in the color palette, or 0 to default to 2^n 
     */
    unsigned biClrUsed;        

    /**
     * @brief  the number of important colors used, or 0 when every color is important; generally ignored 
     */
    unsigned biClrImportant;    
} BITMAPINFOHEADER; 

/**
 * @brief The RGBQUAD  structure describes a color consisting of relative intensities of red, green, and blue. 
 */
typedef struct 
{ 
    unsigned char rgbBlue;      /**< the value of rgbBlue(0~255) */
    unsigned char rgbGreen;     /**< the value of rgbGreen (0~255) */
    unsigned char rgbRed;       /**< the value of rgbRed (0~255) */
    unsigned char rgbReserved;  /**< depeneded on the version of Microsoft's documentation */
} RGBQUAD;


class BMP
{
    public:
        
        /**
         * @brief Alloc space for m_bfile and m_binfo and fill the space with value 0, init the value of every unit in m_bquad,  Eventually make the file pointer m_pf to NULL.
         */
        BMP();

        /**
         * @brief Call fclose() to close the image file stream which m_fp points to.
         */
        ~BMP();
    
    public:
        
        /**
         * @brief Open the file in assigned mode and cause the file position to be repositioned to the beginning of the file.
         *
         * @return the consequence of file open. 0 represents failure and 1 represents success.
         */
        int open(const char* filename,  /**< [in] file name */
                 const char* mode       /**< [in] open mode */
                );

        /**
         * @brief Open the file in default mode and  cause the file position to be repositioned to the beginning of the file.
         *
         * @return the consequence of file open. 0 represents failure and 1 represents success.
         */
        int open(FILE*& file /**< [in] file stream */ );  
    
        /** 
         * @brief Close the image file stream which m_fp points to.
         */
        void close();

        /**
         * @brief Create a BMP image file which file pointer m_fp points to.
         * 
         * Initiate the file header with parameter "width", "height" and other default value. 
         * The part that is initialized includes the general information about the bit map,  the detailed information about the image and the data of the image.
         * 
         * @return true if creates success, or false if creates failed.
         */ 
        bool createBMP(unsigned char* buf,  /**< [in] the data of bit map*/ 
                       int width,           /**< [in] the width of bit map */
                       int height           /**< [in] the height of bit map */
                      );

        /**
         * @brief Create a BMP image file which file pointer m_fp points to.
         * 
         * Initiate the file header with parameter "width", "height" and other default value. 
         * The part that is initialized includes the general information about the bit map,   the detailed information about the image and the data of the image.
         * The float type data of image will be converted by calling function DataConvert().
         * 
         * @return true if creates success, or false if creates failed.
         */            
        bool createBMP(float* buf,  /**< [in] the data of bit map*/ 
                       int width,   /**< [in] the width of bit map */
                       int height   /**< [in] the height of bit map */
                      );

        /**
         * @brief Create a BMP image file which file pointer m_fp points to.
         * 
         * Initiate the file header with parameter "width", "height" and other default value.
         * The part that is initialized includes the general information about the bit map,  the detailed information about the image and the data of the image.
         * The short type data of image will be converted by calling function DataConvert().
         * 
         * @return true if creates success, or false if creates failed.
         */ 
        bool createBMP(short* buf,  /**< [in] the data of bit map*/
                       int width,   /**< [in] the width of bit map */
                       int height   /**< [in] the height of bit map */
                      );
        
        /**
         * @brief Initiate the content of "BITMAPFILEHEADER" and "BITMAPINFOHEADER"
         */
        bool setHeader(int width,  /**< [in] the width of bit map */
                       int height  /**< [in] the height of bit map */
                      );

        /**
         * @brief Write the data of BMP image, after skipping the file header.
         */
        bool writeIm2D(unsigned char* buf  /**< [in] the data of bit map*/ ); 
                      

    public:

        /**
         * @brief Go back to the beginning of the file stream, then read the file header of bit map into m_bfile and read the info header of bit map into m_binfo
         */
        void readInHeader();

        /**
         * @brief Read in BITMAPFILEHEADER and BITMAPINFOHEADER.
         */
        unsigned getWidth() const;

        /**
         * @brief Get the height of bit map. Must be after readInHeader().
         */
        unsigned getHeight() const;

        /**
         * @brief Get the bit count. Must be after readInHeader().
         */
        unsigned short getBitCount() const;

        /**
         * @brief Get the size of header. Must be after readInHeader().
         */
        unsigned getHeaderSize() const;

        /**
         * @brief Get the size of image data.
         */
        unsigned getDataSize() const;

    private:

        /**
         * @brief  Get the minimum and the maximal value in array buf.
         */
        void MinMax(float* buf,  /**< [in] the data of bit map */
                    int size,    /**< [in] the size of data, that is m_binfo.biSize */
                    float& min,  /**< [out] the minimum value in float type array buf */
                    float& max   /**< [out] the maximal value in float type array buf */
                   );
        
        /**
         * @brief  Get the minimum and the maximal value in array buf.
         */
        void MinMax(short* buf,  /**< [in] the data of bit map */
                    int size,    /**< [in] the size of data, that is m_binfo.biSize */
                    float& min,  /**< [out] the minimum value in short type array buf */
                    float& max   /**< [out] the maximal value in short type array buf */
                   );
        
        /**
         * @brief Convert data in float type into char type.
         */
        void DataConvert(unsigned char* dst,  /**< [out] the converted data */
                         float* src,          /**< [in] data waiting to be converted */
                         int size             /**< [in] size of data  */
                        );

        /**
         * @brief Convert data in short type into char type.
         */
        void DataConvert(unsigned char* dst,  /**< [out] the converted data */
                         short* src,          /**< [in] data waiting to be converted */
                         int size             /**< [in] size of data  */
                        );

    private:

        BITMAPFILEHEADER m_bfile;  /**< the general information of the bit map */
        BITMAPINFOHEADER m_binfo;  /**< the detailed information of the image */
        RGBQUAD m_bquad[256];      /**< 256 kinds of color */
    
        FILE* m_fp;                /**< pointer to the image file */
}; 

#endif // BMP_H
