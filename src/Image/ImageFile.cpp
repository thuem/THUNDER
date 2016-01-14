/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description: a image file class
 *
 * Manual:
 * ****************************************************************************/

#include "ImageFile.h"

ImageFile::ImageFile() {}

ImageFile::ImageFile(const char* filename,
                     const char* option)
{
    _file = fopen(filename, option);
}

ImageFile::~ImageFile()
{
    clear();
}

void ImageFile::display() const
{
    printf("Num of Columns:               %6d\n", nCol());
    printf("Num of Rows:                  %6d\n", nRow());
    printf("Num of Slices(Frames):        %6d\n", nSlc());
    printf("Mode:                         %6d\n", mode());
    printf("Symmetry Data Size:           %6d\n", symmetryDataSize());

    if (_MRCHeader != NULL)
    {
        printf("\n");
        printf("MRC Header Size:              %6d\n", sizeof(MRCHeader));
        printf("Num of Intervals along Column:%6d\n", _MRCHeader->mx);
        printf("Num of Intervals along Row:   %6d\n", _MRCHeader->my);
        printf("Num of Intervals along Slice: %6d\n", _MRCHeader->mz);
        printf("Cell Dimension (AA, Column):  %6.3f\n", _MRCHeader->cella[0]);
        printf("Cell Dimension (AA, Row):     %6.3f\n", _MRCHeader->cella[1]);
        printf("Cell Dimension (AA, Slice):   %6.3f\n", _MRCHeader->cella[2]);
        printf("Cell Angle (AA, Column):      %6.3f\n", _MRCHeader->cellb[0]);
        printf("Cell Angle (AA, Row):         %6.3f\n", _MRCHeader->cellb[1]);
        printf("Cell Angle (AA, Slice):       %6.3f\n", _MRCHeader->cellb[2]);
        printf("Axis along Column:            %6d\n", _MRCHeader->mapc);
        printf("Axis along Row:               %6d\n", _MRCHeader->mapr);
        printf("Axis along Slice:             %6d\n", _MRCHeader->maps);
        printf("Space Group:                  %6d\n", _MRCHeader->ispg);
        printf("Orgin (AA, Column):           %6.3f\n", _MRCHeader->origin[0]);
        printf("Orgin (AA, Row):              %6.3f\n", _MRCHeader->origin[1]);
        printf("Orgin (AA, Slice):            %6.3f\n", _MRCHeader->origin[2]);
    }
}

int ImageFile::mode() const {return _metaData.mode;}

int ImageFile::nCol() const {return _metaData.nCol;}

int ImageFile::nRow() const {return _metaData.nRow;}

int ImageFile::nSlc() const {return _metaData.nSlc;}

int ImageFile::size() const
{
    return _metaData.nCol * _metaData.nRow * _metaData.nSlc;
}

int ImageFile::symmetryDataSize() const
{
    return _metaData.symmetryDataSize;
}

void ImageFile::setSize(const int nCol,
                        const int nRow,
                        const int nSlc)
{
    _metaData.nCol = nCol;
    _metaData.nRow = nRow;
    _metaData.nSlc = nSlc;
}

void ImageFile::readMetaData()
{
    readInMetaDataMRC();
}

void ImageFile::readMetaData(const Image& src)
{
    setSize(src.nColRL(), src.nRowRL());
    _metaData.mode = 2;
}

void ImageFile::readMetaData(const Volume& src)
{
    setSize(src.nColRL(), src.nRowRL(), src.nSlcRL());
    _metaData.mode = 2;
}

void ImageFile::readImage(Image& dst,
                          const int iSlc,
                          const char* fileType)
{
    if (strcmp(fileType, "MRC") == 0)
    {
        if (iSlc < 0 || iSlc >= nSlc())
            REPORT_ERROR("Index of slice is out boundary.");
        readImageMRC(dst, iSlc);
    }
    else if (strcmp(fileType, "BMP") == 0)
    {
        if (iSlc != 0)
            REPORT_ERROR("When read in a BMP file, iSlc must be 0.");
        readInImageBMP(dst);
    }
    else
        REPORT_ERROR("File type can not be recognized");
}

void ImageFile::readVolume(Volume& dst,
                           const char* fileType)
{
    readVolumeMRC(dst);
}

void ImageFile::writeImage(const ImageBase& src,
                           const char* filename)
{
    writeImageMRC(src, filename);
}

void ImageFile::clear()
{
    if (_file != NULL) 
    {
        fclose(_file);
        _file = NULL;
    }

    if (_symmetryData != NULL)
    {
        delete[] _symmetryData;
        _symmetryData = NULL;
    }

    if (_MRCHeader != NULL)
    {
        delete _MRCHeader;
        _MRCHeader = NULL;
    }

    reset();
}

void ImageFile::readInSymmetryData()
{
    if (symmetryDataSize() != 0)
    {
        if (fseek(_file, 1024, 0) != 0)
            REPORT_ERROR("Fail to read in an image.");

        _symmetryData = new char[symmetryDataSize()];

        if (fread(_symmetryData, 1, symmetryDataSize(), _file) == 0)
            REPORT_ERROR("Fail to read in an image.");
    }
}

void ImageFile::readInImageMRC(Image& image, int iSlc)
{
    readInSymmetryData();

    image.clear();
	image.setSize(nCol(), nRow());
    size_t imageSize = image.totalSize();

    // from the beginning the this file, search for the start of
    // actual image data; and check whether an error occurs or not 
    if (fseek(_file, 1024 + symmetryDataSize() + imageSize * iSlc, 0) != 0)
        REPORT_ERROR("Fail to read in an image.");
	
    switch (mode())
    {
        case 0:
        {
            char* unCast= new char[imageSize];
			if (fread(unCast, 1,
                        imageSize * transformModeToByte(mode()), _file) == 0)
                REPORT_ERROR("Fail to read in an image.");
            for (size_t i = 0; i < imageSize; i++)
            {
                image.getData()[i] = (float)unCast[i];
            }
            delete[] unCast;
            break;
        }
        case 1:
        {
            short* unCast = new short[imageSize];
			if (fread(unCast, 1,
                        imageSize * transformModeToByte(mode()), _file) == 0)
                REPORT_ERROR("Fail to read in an image.");
            for (size_t i = 0; i < imageSize; i++)
            {
                image.getData()[i] = (float)unCast[i];
            }
            delete[] unCast;
            break;
        }
        case 2:
        {
            if (fread(image.getData(), 1,
                        imageSize * transformModeToByte(mode()), _file) == 0)
                REPORT_ERROR("Fail to read in an image");
            break;
        }
    }
}

void ImageFile::readInImageBMP(Image& image)
{
    try
    {
        BMP bmp;
        bmp.open(_file);
        bmp.readInHeader();
        
        std::cout << "Width = " << bmp.getWidth() << std::endl;
        std::cout << "Height = " << bmp.getHeight() << std::endl;
        std::cout << "BitCount = " << bmp.getBitCount() << std::endl;
        std::cout << "HeaderSize = " << bmp.getHeaderSize() << std::endl;
        std::cout << "DataSize = " << bmp.getDataSize() << std::endl;
    
        image.clear();
        image.setSize(bmp.getWidth(), bmp.getHeight());

        rewind(_file);
        fseek(_file, bmp.getHeaderSize(), 0);
        if (bmp.getBitCount() == 8)
        {
            unsigned char* bmpData = new unsigned char[bmp.getDataSize()];
            if (fread(bmpData, 1, bmp.getDataSize(), _file)
                    < bmp.getDataSize())
                REPORT_ERROR("Fail to read in an BMP file.");
            for (int i = 0; i < bmp.getDataSize(); i++)
                image.getData()[i] = (float)bmpData[i];
            delete[] bmpData; // release the allocated memory
        }
        else
            REPORT_ERROR("Unsupported BMP coding mode.");
    }
    catch (Error& error)
    {
        std::cout << error;
    }
}

void ImageFile::readInVolumeMRC(Volume& volume)
{
    readInSymmetryData();

    volume.clear();
	volume.setSize(nCol(), nRow(), nSlc());
    size_t size = volume.totalSize();

    // from the beginning the this file, search for the start of
    // actual image data; and check whether an error occurs or not 
    if (fseek(_file, 1024 + symmetryDataSize(), 0) != 0)
        REPORT_ERROR("Fail to read in an volume.");
	
    switch (mode())
    {
        case 0:
        {
            char* unCast= new char[size];
			if (fread(unCast, 1,
                      size * transformModeToByte(mode()), _file) == 0)
                REPORT_ERROR("Fail to read in an image.");
            for (size_t i = 0; i < size; i++)
            {
                volume.getData()[i] = (float)unCast[i];
            }
            delete[] unCast;
            break;
        }
        case 1:
        {
            short* unCast = new short[size];
			if (fread(unCast, 1,
                      size * transformModeToByte(mode()), _file) == 0)
                REPORT_ERROR("Fail to read in an image.");
            for (size_t i = 0; i < size; i++)
            {
                volume.getData()[i] = (float)unCast[i];
            }
            delete[] unCast;
            break;
        }
        case 2:
        {
            if (fread(volume.getData(), 1,
                      size * transformModeToByte(mode()), _file) == 0)
                REPORT_ERROR("Fail to read in an image");
            break;
        }
    }
}

void ImageFile::readInMetaDataMRC()
{
    try
    {
        if (_file == 0) // the file point should not be null
            REPORT_ERROR("Can not read in a file which does not exist.");
    
        rewind(_file);
        // reset the location where the file pointer points to
        // to the head of this file

        // allocate _MRCHeader
        _MRCHeader = new MRCHeader;

        // read MRCHeader into _MRCHeader
        int headSize = fread(_MRCHeader, 1, 1024, _file);
        if (headSize != 1024)
            REPORT_ERROR("Fail to read in mrc header file.");

        // set mode (what data type the image is)
        _metaData.mode = _MRCHeader->mode;
        
        if ((_metaData.mode < 0) ||
            (_metaData.mode > 6))
        {
            WARNING("Invalid Mode. Set to default value 2.");
            _metaData.mode = 2;
        }

        _metaData.nCol = _MRCHeader->nx;
        _metaData.nRow = _MRCHeader->ny;
        _metaData.nSlc = _MRCHeader->nz;

        _metaData.symmetryDataSize = _MRCHeader->nsymbt;
    }
    catch (Error& error)
    {
        std::cout << error;
    }
}

void ImageFile::writeOutImageMRC(ImageBase& image, const char* filename)
{
    FILE* file = fopen(filename, "w");

    MRCHeader header;
    header.mode = _metaData.mode;
    header.nx = _metaData.nCol;
    header.ny = _metaData.nRow;
    header.nz = _metaData.nSlc;
    header.mx = _metaData.nCol;
    header.my = _metaData.nRow;
    header.mz = _metaData.nSlc;
    header.cella[0] = 1;
    header.cella[1] = 1;
    header.cella[2] = 1;
    header.cellb[0] = 90;
    header.cellb[1] = 90;
    header.cellb[2] = 90;
    header.nsymbt = _metaData.symmetryDataSize;
    header.dmin = image.min();
    header.dmax = image.max();
    header.dmean = image.mean();

    rewind(file);
    if (fwrite(&header, 1, 1024, file) == 0 || (symmetryDataSize() != 0 &&
        fwrite(_symmetryData, 1, symmetryDataSize(), file) == 0) ||
        fwrite(image.getData(), 1, image.totalSize() * 4, file) == 0)
        REPORT_ERROR("Fail to write out an image.");

    fclose(file);
}
