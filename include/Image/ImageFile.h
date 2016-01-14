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

class ImageFile
{
    private:

        FILE* _file = NULL:

        ImageMetaData _metaData;

        char* _symmetryData = NULL;

        MRCHeader* _MRCHeader = NULL;

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

        void writeImage(const ImageBase& src,
                        const char* filename);

        void clear();

    private:

        void readMetaDataMRC();

        void readSymmetryData();

        void readImageMRC(Image& src,
                          const int iSlc = 0);

        void readImageBMP(Image& src);

        void readVolumeMRC(Volume& src);

        void writeImageMRC(const ImageBase& dst,
                           const char* filename);
};

#endif // IMAGE_FILE_H
