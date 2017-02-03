/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef IMAGE_META_DATA_H
#define IMAGE_META_DATA_H

struct ImageMetaData
{
    int nCol; // number of columns
    int nRow; // number of rows
    int nSlc; // number of slices

    int mode; // data type of this image
    // 0: image, signed 8-bit byte ranges from -128 to 127
    // 1: image, 16-bit
    // 2: image, 32-bit
    // 3: transform, complex, 16-bit integers
    // 4: transform, complex, 32-bit reals
    // 6: image, unsigned 16-bit bytes, ranges from 0 to 65535

    int symmetryDataSize; // number of bytes to store symmetry data

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


inline int  BYTE_MODE(const int _mode) 
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

        FILE* _file;

        ImageMetaData _metaData;

        char* _symmetryData;

        MRCHeader _MRCHeader;

    public:

        ImageFile();

        ImageFile(const char* filename,
                  const char* option);

        ~ImageFile();
        
        void display() const;

        int mode() const;

        int nCol() const;

        int nRow() const;

        int nSlc() const;

        int size() const;

        int symmetryDataSize() const;

        void setSize(const int nCol,
                     const int nRow,
                     const int nSlc = 1);
        
        void readMetaData();

        void readMetaData(const Image& src);

        void readMetaData(const Volume& src);

        void readImage(Image& dst,
                       const int iSlc = 0,
                       const char* fileType = "MRC");

        void readVolume(Volume& dst,
                        const char* fileType = "MRC");

        void writeImage(const char dst[],
                        const Image& src,
                        const double pixelSize = 1);

        void writeVolume(const char dst[],
                         const Volume& src,
                         const double pixelSize = 1);

        void clear();

    private:

        void readMetaDataMRC();

        void fillMRCHeader(MRCHeader& header) const;        

        void readSymmetryData();

        void readImageMRC(Image& dst,
                          const int iSlc = 0);

        void readImageBMP(Image& dst);

        void readVolumeMRC(Volume& dst);

        void writeImageMRC(const char dst[],
                           const Image& src,
                           const double pixelSize);

        void writeVolumeMRC(const char dst[],
                            const Volume& src,
                            const double pixelSize);
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
              = (double)unCast[MESH_IMAGE_INDEX(i, \
                                                j, \
                                                dst.nColRL(), \
                                                dst.nRowRL())]; \
        delete[] unCast; \
    }()
*/


template <typename T> inline void  IMAGE_READ_CAST
                      (FILE * imFile, 
                       Image  &dst )
{ 
        T * unCast = new T[dst.sizeRL()]; 
        if (fread(unCast, sizeof(T), dst.sizeRL()  ,imFile) == 0) 
            REPORT_ERROR("Fail to read in an image."); 
        for (int j = 0; j < dst.nRowRL(); j++) 
            for (int i = 0; i < dst.nColRL(); i++) 
                dst(IMAGE_INDEX(i, j, dst.nColRL())) 
              = (double)unCast[MESH_IMAGE_INDEX(i, 
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
                  = (double)unCast[MESH_VOLUME_INDEX(i, \
                                                     j, \
                                                     k, \
                                                     dst.nColRL(), \
                                                     dst.nRowRL(), \
                                                     dst.nSlcRL())];  \
        delete[] unCast; \
    }()

*/

template <typename T> inline void  VOLUME_READ_CAST 
                                  ( FILE *imFile, 
                                    Volume &dst
                                    ) 
{ 
        T* unCast = new T[dst.sizeRL() ]; 
        if (fread(unCast, sizeof(T), dst.sizeRL() , imFile) == 0) 
            REPORT_ERROR("Fail to read in an image."); 
        for (int k = 0; k < dst.nSlcRL(); k++) 
            for (int j = 0; j < dst.nRowRL(); j++) 
                for (int i = 0; i < dst.nColRL(); i++) 
                    dst(VOLUME_INDEX(i, j, k, dst.nColRL(), dst.nRowRL())) 
                  = (double)unCast[MESH_VOLUME_INDEX(i, 
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

template <typename T> inline void  IMAGE_WRITE_CAST 
                                  ( FILE *imFile, 
                                   const Image  &src  
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

template <typename T>  inline void VOLUME_WRITE_CAST(FILE *imFile,const  Volume &src)    
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
