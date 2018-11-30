
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
 *  Fande Yu  | 2018/09/22 | 1.4.11.080913 | annotation
 * 
 *
 *  @brief ImageFile.h contains the method of reading and writing different types of image. 
 *  Image types includs "BMP", MRC","Volume" and "Image". It also contains operations on different file header.
 *  Data type cast is one of the most important part in this head file. 
 */
#ifndef IMAGE_META_DATA_H
#define IMAGE_META_DATA_H

struct ImageMetaData
{
    int nCol;  /**< number of columns */
    int nRow;  /**< number of rows */
    int nSlc;  /**< number of slices */

    /**
     * @brief data type of this image
     *
     * 0: image, signed 8-bit byte ranges from -128 to 127
     * 1: image, 16-bit
     * 2: image, 32-bit
     * 3: transform, complex, 16-bit integers
     * 4: transform, complex, 32-bit reals
     * 6: image, unsigned 16-bit bytes, ranges from 0 to 65535
     */
    int mode;
    int symmetryDataSize; /**< number of bytes to store symmetry data */

    /**
     * @brief Initiate the object of ImageMetaData, default data type of image is 32-bit.
     */
    ImageMetaData()
    {
        nCol = 0;
        nRow = 0;
        nSlc = 0;
        mode = 2;
        symmetryDataSize = 0;
    }
};

#endif // IMAGE_META_DATA_H

#ifndef IMAGE_FILE_H
#define IMAGE_FILE_H

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "Logging.h"
#include "Precision.h"

#include "MRCHeader.h"
#include "Image.h"
#include "Volume.h"
#include "BMP.h"

#define DEBUGWRITEIMAGE

/*
#define BYTE_MODE(mode) \
    [](const int _mode) \
    { \
        switch (_mode) \
        { \
            case 0: return 1; \
            case 1: return 2; \
            case 2: return 4; \
            case 3: return 4; \
            case 4: return 8; \
            case 6: return 2; \
            default: return -1; \
        } \
    }(mode)
*/

/**
 * @brief Get the number of byte of related mode.
 *
 * @return the number of byte of related mode.
 */
inline int  BYTE_MODE(const int _mode  /**< [in] image mode in struct ImageMetaData */ )
{

        switch (_mode) 
        { 
            case 0: return 1; 
            case 1: return 2; 
            case 2: return 4; 
            case 3: return 4; 
            case 4: return 8; 
            case 6: return 2; 
            default: return -1;
        } 
};

class ImageFile
{
    private:

        FILE* _file;              /**< file pointer to the image file */

        ImageMetaData _metaData;  /**< meta data of the image */

        char* _symmetryData;      /**< pointer to symmetry data */

        MRCHeader _MRCHeader;     /**< MRC format header */

    public:

        /**
         * @brief The default no-parameter constructor of class ImageFilei.
         *
         * Intiate the pointers _file and  _symmetr to NULL.
         */
        ImageFile();

        /**
         * @brief The constructor with parameter of class ImageFile
         * 
         * Intiate the pointer _file to the image file, and pointer _symmetr to NULL.
         * If the file does not exist, record then abort.
         */
        ImageFile(const char* filename,  /**< [in] file path */
                  const char* option     /**< [in] the mode of fopen */
                  );  

        /**
         * @brief Deconstructor of class ImageFile.
         * 
         * Call function clear() to do clear work. 
         */
        ~ImageFile();
        
        /**
         * @brief Show the information of the object which calls the function.
         * 
         * Including MRC format information.
         */
        void display() const;

        /**
         * @brief Get the data type of the image.
         * 
         * @return the data type of the image.
         */
        int mode() const;

        /**
         * @brief Get the number of columns of the image.
         * 
         * @return the number of columns of the image.
         */
        int nCol() const;

        /**
         * @brief Get the number of rows of the image.
         * 
         * @return the number of rows of the image.
         */
        int nRow() const;

        /**
         * @brief Get the number of slices of the image.
         * 
         * @return the number of slices of the image.
         */
        int nSlc() const;

        /**
         * @brief Get the size of the image.
         * 
         * @return the size of the image, that is nCols * nRows * nSlc.
         */
        int size() const;

        /**
         * @brief Get the size of symmertry data.
         * 
         * @return the size of symmetry data, that is _metaData.symmertrtData .
         */
        int symmetryDataSize() const;

        /**
         * @brief Set the parameter in _metaData.
         * 
         * The default value of nSlc is 1.
         */
        void setSize(const int nCol,     /**< [in] the number of columns */
                     const int nRow,     /**< [in] the number of rows */
                     const int nSlc = 1  /**< [in] the number of slices, default in 1 */
                    );
        
        /**
         * @brief Read meta data without any parameter.
         * 
         * Calling function readMetaDataMRC().
         */
        void readMetaData();

        /**
         * @brief Read meta data from src while the type of src is Image.
         */
        void readMetaData(const Image& src  /**< [in] source image */ );  

        /**
         * @brief Read meta data from src while the type of src is Volume.
         */
        void readMetaData(const Volume& src  /**< [in] source volume image */); 

        /**
         * @brief Read image according to fileType.
         * 
         * Call function readImageMRC() or readImageBMP() according to the fileType, or report error if fileType is not one of MRC and BMP.
         */
        void readImage(Image& dst,                   /**< [out] destination image */        
                       const int iSlc = 0,           /**< [in] used to calculate the size of data that is skipped, default is 0 */
                       const char* fileType = "MRC"  /**< [in] file type, default is "MRC */
                      );

        /**
         * @brief Read Volume file according to fileType
         * 
         * Call function readVolumeMRC().
         */
        void readVolume(Volume& dst,                    /**< [out] destination image */ 
                        const char* fileType = "MRC"    /**< [in] file type, default is "MRC" */
                       );

        /**
         * @brief Write data into dst from src
         * 
         * Call function writeImageMRC().
         */
        void writeImage(const char dst[],           /**< [out] path of destination file */
                        const Image& src,           /**< [in] source data */
                        const RFLOAT pixelSize = 1  /**< [in] pixel size,default is 1 */
                       );

        /**
         * @brief  Write data into dst from src
         * 
         * Call function WriteVolumeMRC().
         */
        void writeVolume(const char dst[],           /**< [out] path of destination file */
                         const Volume& src,          /**< [in] source data */
                         const RFLOAT pixelSize = 1  /**< [in] pixel size, default is 1 */
                        );

        /**
         * @brief Initiate some member varaible and MRCHeader, then write header and symmetry data into _file
         */
        void openStack(const char dst[],       /**< [out] path of destination file */
                       const int size,         /**< [in] the number of columns and rows in _metaData */
                       const int nSlc,         /**< [in] the number of slices in _metaData */
                       const RFLOAT pixelSize  /**< [in] pixel size, default is 1 */
                      );

        /**
         * @brief Skip the header and writeiamge dat a into _file.
         * 
         * Call template IMAGE_WRITE_CAST to cast data type from float to RFLOAT.
         */
        void writeStack(const Image& src,  /**< [in] data source */
                        const int iSlc     /**< [in] used to calculate the size of data that is skipped */
                       );

        /**
         * @brief Close the file stream which _file points to.
         */
        void closeStack();

        /**
         * @brief Clear the pointers and alloced space
         * 
         * Close the file stream which pointer _file points to and free the space alloced in _symmertryData.
         */
        void clear();

    private:

        /**
         * @brief Read data from MRCHeader into _metaData.
         * 
         * Read the forefront 1024 bytes of the MRC file, get the requisite information including the mode, the number of columns, rows, slices, and symmetry data size. 
         * Then store them into _metaData.
         * If failed to read MRCHeader, report error and abort.
         */
        void readMetaDataMRC();

        /**
         * @brief Fill the incoming header.
         * 
         * First call memset to flush the incoming header to clear its default value, then assign value in header with values in _metaData and some default values.
         */
        void fillMRCHeader(MRCHeader& header  /**< [out] header waiting to be filled */) const;  

        /**
         * @brief Read symmetry data into space which pointer _symmetryData points to.
         * 
         * Alloc space according to the value of symmetryDataSize, then read symmetry data in file which pointer _file points to.
         * If failed to read image, report error and abort.
         * If failed to read symmetry data, report error and abort.
         */
        void readSymmetryData();

        /**
         * @brief Read MRC image into dst.
         * 
         * Call function alloc() in dst, then skip head data, finally read data in _file into dst. 
         * Because the data type of image which _file points to is not RFLOAT, cast is requisite. 
         */
        void readImageMRC(Image& dst,         /**< [out] destination image*/
                          const int iSlc = 0  /**< [in] used to calculate the size of data that is skipped, default is 0 */
                         );

        /**
         * @brief Read BMP image into dst.
         * 
         * Call function alloc() in dst, then skip head data, finally read data in _file into dst. 
         * If the bitcount of the BMP image is 8, cast is needed.
         * Report error if meeting unsupported BMP coding mode.
         */
        void readImageBMP(Image& dst  /**< [out] destination image */);  


        /**
         * @brief Read Volume image into dst.
         * 
         * Call function alloc() in dst, then skip head data, finally read data in _file into dst. 
         * Because the data type of Vlolume which _file points to is not RFLOAT, cast is requisite. 
         */
        void readVolumeMRC(Volume& dst  /**< [out] destination image */);  


        /**
         * @brief Write data into dst from src
         * 
         * Fill MRC header, dispose cell dimensions in angstroms, write header and symmetry data into file, cast image data then write into file.
         */
        void writeImageMRC(const char dst[],       /**< [out] path of destination file */
                           const Image& src,       /**< [in] source data */
                           const RFLOAT pixelSize  /**< [in] pixel size */
                          );

        /**
         * @brief Write data into dst from src
         * 
         * Fill volume MRC header, dispose cell dimensions in angstroms, write header and symmetry data into file, cast volume image data then write into file.
         */
        void writeVolumeMRC(const char dst[],       /**< [out] path of destination file */
                            const Volume& src,      /**< [in] source data */
                            const RFLOAT pixelSize  /**< [in] pixel size */
                           );
};


#define SKIP_HEAD(i) \
    if (fseek(_file, 1024 + symmetryDataSize() + i, 0) != 0) \
        REPORT_ERROR("Fail to read in an image.");


#define IMAGE_INDEX(i, j, nCol) \
    j * (nCol) + i

#define VOLUME_INDEX(i, j, k, nCol, nRow) \
    k * (nCol) * (nRow) + j * (nCol) + i

#define MESH_IMAGE_INDEX(i, j, nCol, nRow) \
    (j + (nRow) / 2) % (nRow) * (nCol) + (i + (nCol) / 2) % (nCol)

#define MESH_VOLUME_INDEX(i, j, k, nCol, nRow, nSlc) \
    (k + (nSlc) / 2) % (nSlc) * (nRow) * (nCol) \
  + (j + (nRow) / 2) % (nRow) * (nCol) \
  + (i + (nCol) / 2) % (nCol)




/*
#define IMAGE_READ_CAST(dst, type) \
    [this, &dst]() \
    { \
        type* unCast = new type[dst.sizeRL()]; \
        if (fread(unCast, 1, dst.sizeRL() * sizeof(type), _file) == 0) \
            REPORT_ERROR("Fail to read in an image."); \
        for (int j = 0; j < dst.nRowRL(); j++) \
            for (int i = 0; i < dst.nColRL(); i++) \
                dst(IMAGE_INDEX(i, j, dst.nColRL())) \
              = (RFLOAT)unCast[MESH_IMAGE_INDEX(i, \
                                                j, \
                                                dst.nColRL(), \
                                                dst.nRowRL())]; \
        delete[] unCast; \
    }()
*/

/**
 * @brief Cast data in Image object into RFLOAT from different data type while being read into dst.
 * 
 * Because the data type of image storage depends on mode, including "char", "short", "float".
 * They have to be casted into RFLOAT for following processing.
 */
template <typename T> inline void  IMAGE_READ_CAST (FILE * imFile,  /**< [in] file waiting to be read */
                                                    Image  &dst     /**< [out] Image object to store MRCImage */
                                                   )
{ 
        T * unCast = new T[dst.sizeRL()]; 
        if (fread(unCast, sizeof(T), dst.sizeRL()  ,imFile) == 0) 
            REPORT_ERROR("Fail to read in an image."); 
        for (int j = 0; j < dst.nRowRL(); j++) 
            for (int i = 0; i < dst.nColRL(); i++) 
                dst(IMAGE_INDEX(i, j, dst.nColRL())) 
              = (RFLOAT)unCast[MESH_IMAGE_INDEX(i, 
                                                j, 
                                                dst.nColRL(), 
                                                dst.nRowRL())]; 
        delete[] unCast; 
}

/*
#define VOLUME_READ_CAST(dst, type) \
    [this, &dst]() \
    { \
        type* unCast = new type[dst.sizeRL()]; \
        if (fread(unCast, 1, dst.sizeRL() * sizeof(type), _file) == 0) \
            REPORT_ERROR("Fail to read in an image."); \
        for (int k = 0; k < dst.nSlcRL(); k++) \
            for (int j = 0; j < dst.nRowRL(); j++) \
                for (int i = 0; i < dst.nColRL(); i++) \
                    dst(VOLUME_INDEX(i, j, k, dst.nColRL(), dst.nRowRL())) \
                  = (RFLOAT)unCast[MESH_VOLUME_INDEX(i, \
                                                     j, \
                                                     k, \
                                                     dst.nColRL(), \
                                                     dst.nRowRL(), \
                                                     dst.nSlcRL())];  \
        delete[] unCast; \
    }()

*/
/**
 * @brief Cast data in Volume object into RFLOAT from different data type while being read into dst.
 * 
 * Because the data type of Volume storage depends on mode, including "char", "short", "float".
 * They have to be casted into RFLOAT for following processing.
 */
template <typename T> inline void  VOLUME_READ_CAST(FILE *imFile,  /**< [in] file waiting to be read */
                                                    Volume &dst    /**< [out] Volume object to store volume image */
                                                   ) 
{ 
        T* unCast = new T[dst.sizeRL() ]; 
        if (fread(unCast, sizeof(T), dst.sizeRL() , imFile) == 0) 
            REPORT_ERROR("Fail to read in an image."); 
        for (int k = 0; k < dst.nSlcRL(); k++) 
            for (int j = 0; j < dst.nRowRL(); j++) 
                for (int i = 0; i < dst.nColRL(); i++) 
                    dst(VOLUME_INDEX(i, j, k, dst.nColRL(), dst.nRowRL())) 
                  = (RFLOAT)unCast[MESH_VOLUME_INDEX(i, 
                                                     j, 
                                                     k, 
                                                     dst.nColRL(), 
                                                     dst.nRowRL(), 
                                                     dst.nSlcRL())];  
        delete[] unCast; 
}

/*
#define IMAGE_WRITE_CAST(src, type) \
    [this, &src]() \
    { \
        type* cast = new type[src.sizeRL()]; \
        for (int j = 0; j < src.nRowRL(); j++) \
            for (int i = 0; i < src.nColRL(); i++) \
                cast[IMAGE_INDEX(i, j, src.nColRL())] \
              = (type)src.iGetRL(MESH_IMAGE_INDEX(i, \
                                                  j, \
                                                  src.nColRL(), \
                                                  src.nRowRL())); \
        if (fwrite(cast, 1, src.sizeRL() * sizeof(type), _file) == 0) \
            REPORT_ERROR("Fail to write in an image."); \
        delete[] cast; \
    }()
*/

/**
 * @brief Cast data in src and write into the file which imFile points to.
 * 
 * Because the data type of image storage depends on mode, including "char", "short", "float".
 * They have to be casted into RFLOAT for following processing.
 */
template <typename T> inline void  IMAGE_WRITE_CAST 
                                  ( FILE *imFile,     /**< [out] data in src will be written */
                                   const Image  &src  /**< [in] data source */  
                                  )
{ 
        T* cast = new T[src.sizeRL()]; 
        for (int j = 0; j < src.nRowRL(); j++) 
            for (int i = 0; i < src.nColRL(); i++) 
                cast[IMAGE_INDEX(i, j, src.nColRL())] 
                = (T)src.iGetRL(MESH_IMAGE_INDEX(i, 
                                                  j, 
                                                  src.nColRL(), 
                                                  src.nRowRL())); 
        if (fwrite(cast, sizeof(T), src.sizeRL() ,imFile) == 0) 
            REPORT_ERROR("Fail to write in an image."); 
        delete[] cast; 
}


/*
#define VOLUME_WRITE_CAST(src, type) \
    [this, &src]() \
    { \
        type* cast = new type[src.sizeRL()]; \
        for (int k = 0; k < src.nSlcRL(); k++) \
            for (int j = 0; j < src.nRowRL(); j++) \
                for (int i = 0; i < src.nColRL(); i++) \
                    cast[VOLUME_INDEX(i, j, k, src.nColRL(), src.nSlcRL())] \
                  = (type)src.iGetRL(MESH_VOLUME_INDEX(i, \
                                                       j, \
                                                       k, \
                                                       src.nColRL(), \
                                                       src.nRowRL(), \
                                                       src.nSlcRL())); \
        if (fwrite(cast, 1, src.sizeRL() * sizeof(type), _file) == 0) \
            REPORT_ERROR("Fail to write in an image."); \
        delete[] cast; \
    }()
*/



/**
 * @brief Cast data in src and write into the file which imFile points to.
 * 
 * Because the data type of Volume storage depends on mode, including "char", "short", "float".
 * They have to be casted into RFLOAT for following processing.
 */
template <typename T>  inline void VOLUME_WRITE_CAST(FILE *imFile,       /**< [out] data in src will be written */
                                                     const  Volume &src  /**< [in] data source */ 
                                                    )    
{
        T* cast = new T[src.sizeRL()]; 
        for (int k = 0; k < src.nSlcRL(); k++) 
            for (int j = 0; j < src.nRowRL(); j++) 
                for (int i = 0; i < src.nColRL(); i++) 
                    cast[VOLUME_INDEX(i, j, k, src.nColRL(), src.nSlcRL())] 
                  = (T)src.iGetRL(MESH_VOLUME_INDEX(i, 
                                                       j, 
                                                       k, 
                                                       src.nColRL(), 
                                                       src.nRowRL(), 
                                                       src.nSlcRL())); 
        if (fwrite(cast, sizeof(T) , src.sizeRL() , imFile) == 0) 
            REPORT_ERROR("Fail to write in an image."); 
        delete[] cast; 
}

#endif 
// IMAGE_FILE_H
