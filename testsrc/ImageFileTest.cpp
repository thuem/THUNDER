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

#define N 380

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
        
        fft.fw(image);

        sprintf(filename, "Image_%04d_FT.bmp", i);
        image.saveFTToBMP(filename, 0.01);

        fft.bw(image);
    }
}
