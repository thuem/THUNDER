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
    int nCol = 0; // number of columns
    int nRow = 0; // number of rows 
    int nSlc = 0; // number of slices

    int mode = 2; // data type of this image
    // 0: image, signed 8-bit byte ranges from -128 to 127
    // 1: image, 16-bit
    // 2: image, 32-bit
    // 3: transform, complex, 16-bit integers
    // 4: transform, complex, 32-bit reals
    // 6: image, unsigned 16-bit bytes, ranges from 0 to 65535

    int symmetryDataSize = 0; // number of bytes to store symmetry data
};

#endif // IMAGE_META_DATA_H

#ifndef IMAGE_FILE_H
#define IMAGE_FILE_H

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "Error.h"

#include "MRCHeader.h"
#include "Image.h"
#include "Volume.h"
#include "BMP.h"

#define DEBUGWRITEIMAGE

#define BYTE_MODE(mode) \
    [&mode]() \
    { \
        switch (mode) \
        { \
            case 0: return 1; \
            case 1: return 2; \
            case 2: return 4; \
            case 3: return 4; \
            case 4: return 8; \
            case 6: return 2; \
            default: return -1; \
        } \
    }()

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

#define SKIP_HEAD(i) \
    if (fseek(_file, 1024 + symmetryDataSize() + i, 0) != 0) \
        REPORT_ERROR("Fail to read in an image.");

#define CHECK_FILE_VALID \
    if (_file == NULL) \
        REPORT_ERROR("Image file does not exist.");

class ImageFile
{
    private:

        FILE* _file = NULL;

        ImageMetaData _metaData;

        char* _symmetryData = NULL;

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
                        const Image& src);

        void writeVolume(const char dst[],
                         const Volume& src);

        void clear();

    private:

        void fillMRCHeader(MRCHeader& header) const;

        void readMetaDataMRC();

        void readSymmetryData();

        void readImageMRC(Image& dst,
                          const int iSlc = 0);

        void readImageBMP(Image& dst);

        void readVolumeMRC(Volume& dst);

        void writeImageMRC(const char dst[],
                           const Image& src);

        void writeVolumeMRC(const char dst[],
                            const Volume& src);
};

#endif // IMAGE_FILE_H
