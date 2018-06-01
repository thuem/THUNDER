#include "BMP.h"

BMP::BMP()
{
	memset((void*)&m_bfile, 0, sizeof(BITMAPFILEHEADER));
	memset((void*)&m_binfo, 0, sizeof(BITMAPINFOHEADER));
	for (int i = 0; i < 256; i++)
	{
		m_bquad[i].rgbBlue = char(i);
		m_bquad[i].rgbGreen = char(i);
		m_bquad[i].rgbRed = char(i);
		m_bquad[i].rgbReserved = 1;
	}
	m_fp = NULL;
}

BMP::~BMP()
{
    close();
}

int BMP::open(const char* filename, const char* mode)
{
	close();
	m_fp = fopen(filename, mode);
	if (m_fp == NULL) return 0;
	
	rewind(m_fp);
	return 1;
}

int BMP::open(FILE*& file)
{
    close();
    m_fp = file;

    if (m_fp == NULL) return 0;

    rewind(m_fp);
    return 1;
}

void BMP::close()
{
	if (m_fp != NULL) fclose(m_fp);
	m_fp = NULL;
}

void BMP::readInHeader()
{
    rewind(m_fp);

    fread(&m_bfile, sizeof(BITMAPFILEHEADER), 1, m_fp);
    fread(&m_binfo, sizeof(BITMAPINFOHEADER), 1, m_fp);
}

unsigned BMP::getWidth() const
{
    return m_binfo.biWidth;
}

unsigned BMP::getHeight() const
{
    return m_binfo.biHeight;
}

unsigned short BMP::getBitCount() const
{
    return m_binfo.biBitCount;
}

unsigned BMP::getHeaderSize() const
{
	return sizeof(BITMAPFILEHEADER)
           + sizeof(BITMAPINFOHEADER)
           + sizeof(RGBQUAD) * 256;
}

unsigned BMP::getDataSize() const
{
    return m_binfo.biWidth * m_binfo.biHeight * m_binfo.biBitCount / 8;
}

bool BMP::createBMP(unsigned char* buf, int width, int height)
{
	if (buf == NULL) return false;
	
	setHeader(width, height); 
    writeIm2D(buf); 
    
    return true;
}

bool BMP::createBMP(float* buf, int width, int height)
{
	if (buf == NULL) return false;
	
	long size = width * height;
	unsigned char* bufc = new unsigned char[size];
	DataConvert(bufc, buf, size);

	setHeader(width, height); 
    writeIm2D(bufc); 
   
    delete[] bufc;
   
    return true;
}

bool BMP::createBMP(short* buf, int width, int height)
{
	if (buf == NULL) return false;
	
	long size = width * height;
	unsigned char* bufc = new unsigned char[size];
	DataConvert(bufc, buf, size);

	setHeader(width, height); 
    writeIm2D(bufc); 
   
    delete[] bufc;
   
    return true;
}

bool BMP::setHeader(int width, int height)
{
    m_bfile.bfType = 0x4D42;
    m_bfile.bfSize = getHeaderSize() + width * height; 
    m_bfile.bfReserved1 = 0;
    m_bfile.bfReserved2 = 0;
    m_bfile.bfoffBits = getHeaderSize();

    m_binfo.biSize = sizeof(BITMAPINFOHEADER);
    m_binfo.biWidth = width;
    m_binfo.biHeight = height;
    m_binfo.biPlanes = 1;
    m_binfo.biBitCount = 8;
    m_binfo.biCompress = 0;
    m_binfo.biSizeImage = getDataSize();
    m_binfo.biXPelsPerMeter = 300;
    m_binfo.biYPelsPerMeter = 300;
    m_binfo.biClrUsed = 256;
    m_binfo.biClrImportant = 256;
    
	rewind(m_fp);

	if (fwrite(&m_bfile, 1, sizeof(BITMAPFILEHEADER), m_fp) < sizeof(BITMAPFILEHEADER) ||
		fwrite(&m_binfo, 1, sizeof(BITMAPINFOHEADER), m_fp) < sizeof(BITMAPINFOHEADER) ||
		fwrite(m_bquad, 256, sizeof(RGBQUAD), m_fp))
        return false;
		
	return true;
}

bool BMP::writeIm2D(unsigned char* buf)
{
	long ImSize = getDataSize();
	long offset = getHeaderSize();
	if (fseek(m_fp, offset, SEEK_SET) != 0)
        return 0;
	return fwrite(buf, 1, ImSize, m_fp);
}

void BMP::MinMax(float* buf, int size, float& min, float& max)
{
	min = max = buf[0];
	int i;
	for (i = 0; i < size; i++)
	{
		if (min > buf[i])
            min = buf[i];
		if (max < buf[i])
            max = buf[i];
	}
}

void BMP::MinMax(short* buf, int size, short& min, short& max)
{
	min = max = buf[0];
	int i;
	for (i = 0;i < size; i++)
	{
		if (min > buf[i]) min = buf[i];
		if (max < buf[i]) max = buf[i];
	}
}

void BMP::DataConvert(unsigned char* dst, float* src, int size)
{
	float min, max;
	MinMax(src, size, min, max);
	
	float scale = 256.0 / (max - min);
	int i;
	short val;
	for (i = 0; i < size; i++) 
	{
		val = (src[i] - min) * scale + 0.5;
		if (val > 255) val = 255;
		if (val < 0) val = 0;
		dst[i] = (unsigned char)(val);
	}
}

void BMP::DataConvert(unsigned char* dst, short* src, int size)
{
	short min,max;
	MinMax(src, size, min, max);
	
	float scale = 256.0 / (max - min);
	int i;
	short val;
	for (i = 0; i < size; i++) 
	{
		val = (src[i] - min) * scale + 0.5;
		if (val > 255) val = 255;
		if (val < 0) val = 0;
		dst[i] = (unsigned char)(val);
	}
}
