/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "FFT.h"
#include "ImageFile.h"
#include "Volume.h"

#include "Spectrum.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    ImageFile imfA(argv[1], "rb");
    ImageFile imfB(argv[2], "rb");

    imfA.readMetaData();
    imfB.readMetaData();

    Volume mapA, mapB;

    imfA.readVolume(mapA);
    imfB.readVolume(mapB);

    int size = mapA.nColRL();

    FFT fft;

    fft.fw(mapA);
    fft.fw(mapB);

    Volume map(size, size, size, FT_SPACE);

    FOR_EACH_PIXEL_FT(map)
        map[i] = (mapA[i] + mapB[i]) / 2;

    double bFactor;

    bFactorEst(bFactor, map);

    cout << "B-Factor = " << bFactor << endl;

    bFactorFilter(map, map, bFactor);
    
    lowPassFilter(map,
                  map,
                  (double)atoi(argv[3]) / size,
                  2.0 / size);

    fft.bw(map);

    ImageFile imf;
    imf.readMetaData(map);
    imf.writeVolume("Reference_Sharpen.mrc", map);

    return 0;
}
