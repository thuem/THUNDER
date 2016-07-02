/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "ImageFile.h"
#include "FFT.h"
#include "Filter.h"

#define N 380

#define PIXEL_SIZE 1.32
#define EW 3.0

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    FFT fft;

    Image image(N, N, RL_SPACE);

    //ImageFile imf("head.mrc", "rb");
    ImageFile imf("test.mrcs", "rb");
    imf.readMetaData();

    imf.display();

    ImageFile outImf;

    char filename[128];
    for (int i = 0; i < imf.nSlc(); i++)
    {
        imf.readImage(image, i);

        sprintf(filename, "Image_%04d.bmp", i);
        image.saveRLToBMP(filename);

        sprintf(filename, "Image_%04d.mrc", i);
        outImf.readMetaData(image);
        outImf.writeImage(filename, image);

        // perform low pass filter with threshold at 8.7 A
        //double thres = PIXEL_SIZE / 8.7;
        double thres = PIXEL_SIZE / 33;
        double ew = EW / N;

        R2R_FT(image, image, lowPassFilter(image, image, thres, ew));

        sprintf(filename, "Image_%04d_LP.bmp", i);
        image.saveRLToBMP(filename);
        
        fft.fw(image);

        sprintf(filename, "Image_%04d_FT.bmp", i);
        image.saveFTToBMP(filename, 0.01);

        fft.bw(image);
    }
}
