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
    ImageFile imf;

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

    cout << "Open FSC File" << endl;

    FILE* file = fopen(argv[5], "r");

    if (file != NULL)
        cout << "Open FSC Succced!" << endl;

    char buf[FILE_LINE_LENGTH];

    int nLine = 0;
    while (fgets(buf, FILE_LINE_LENGTH, file)) nLine++;

    rewind(file);

    cout << "nLine = " << nLine << endl;
    
    vec fsc(nLine / 2 + 1);
    fsc(0) = 1;
    int idx;
    double res;
    double val;
    for (int i = 0; i < nLine; i++)
    {
        fscanf(file, "%d %lf %lf", &idx, &res, &val);
        //cout << val << endl;
        fsc(i / 2) = val;
    }
    
    for (int i = 0; i < fsc.size(); i++)
        cout << i << ", " << fsc(i) << endl;

    cout << "FSC Weighting Fitler" << endl;
    fscWeightingFilter(map, map, fsc);

    fft.bw(map);

    imf.readMetaData(map);
    imf.writeVolume("Reference_FSC_Weighting.mrc", map);

    fft.fw(map);

    double bFactor;

    bFactorEst(bFactor,
               map,
               atoi(argv[3]),
               atoi(argv[4]));

    // bFactor = -111;

    cout << "B-Factor = " << bFactor << endl;

    bFactorFilter(map, map, bFactor);
    
    lowPassFilter(map,
                  map,
                  (double)atoi(argv[3]) / size,
                  2.0 / size);

    fft.bw(map);

    imf.readMetaData(map);
    imf.writeVolume("Reference_Sharpen.mrc", map);

    return 0;
}
