#ifndef BMP_H
#define BMP_H

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>

typedef struct 
{
    unsigned short bfType; 
    unsigned bfSize;
    unsigned short bfReserved1; 
    unsigned short bfReserved2; 
    unsigned bfoffBits;
} __attribute__ ((packed)) BITMAPFILEHEADER; 

typedef struct 
{ 
    unsigned biSize; 
    unsigned biWidth; 
    unsigned biHeight; 
    unsigned short biPlanes; 
    unsigned short biBitCount; 
    unsigned biCompress; 
    unsigned biSizeImage; 
    unsigned biXPelsPerMeter; 
    unsigned biYPelsPerMeter; 
    unsigned biClrUsed; 
    unsigned biClrImportant; 
} BITMAPINFOHEADER; 

typedef struct 
{ 
    unsigned char rgbBlue; 
    unsigned char rgbGreen; 
    unsigned char rgbRed; 
    unsigned char rgbReserved; 
} RGBQUAD;

class BMP
{
    public:
	    
        BMP();
	    ~BMP();
	
    public:
	    
        int open(const char* filename, const char* mode);
        int open(FILE*& file);
	    void close();
	    bool createBMP(unsigned char* buf, int width, int height);
	    bool createBMP(float* buf, int width, int height);
	    bool createBMP(short* buf, int width, int height);
        
	    bool setHeader(int width, int height);
	    bool writeIm2D(unsigned char* buf);

    public:

        void readInHeader();
        // read in BITMAPFILEHEADER and BITMAPINFOHEADER
        unsigned getWidth() const;
        // must be after readInHeader()
        unsigned getHeight() const;
        // must be after readInHeader()
        unsigned short getBitCount() const;
        // must be after readInHeader();
	    unsigned getHeaderSize() const;
	    unsigned getDataSize() const;

    private:

	    void MinMax(float* buf, int size, float& min, float& max);
	    void MinMax(short* buf, int size, short& min, short& max);
	    void DataConvert(unsigned char* dst, float* src, int size);
	    void DataConvert(unsigned char* dst, short* src, int size);

    private:

	    BITMAPFILEHEADER m_bfile;
	    BITMAPINFOHEADER m_binfo;
	    RGBQUAD m_bquad[256]; 
	
	    FILE* m_fp;
}; 

#endif // BMP_H
