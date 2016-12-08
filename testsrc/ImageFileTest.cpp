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



INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    Volume vol;

    ImageFile imf(argv[1], "rb");
    imf.readMetaData();
    imf.display();
    imf.readVolume(vol);

    ImageFile omf;
    omf.readMetaData(vol);
    omf.writeVolume("out.mrc", vol);
}
