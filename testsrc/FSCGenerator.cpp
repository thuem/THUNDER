/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "ImageFile.h"
#include "Volume.h"
#include "ImageFunctions.h"
#include "Spectrum.h"
#include "FFT.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    ImageFile refAImf(argv[1], "rb");
    ImageFile refBImf(argv[2], "rb");

    refAImf.readMetaData();
    refBImf.readMetaData();

    int N = refAImf.nCol();

    if ((N != refAImf.nRow()) ||
        (N != refAImf.nSlc()) ||
        (N != refBImf.nCol()) ||
        (N != refBImf.nRow()) ||
        (N != refBImf.nSlc()))
        CLOG(FATAL, "LOGGER_SYS") << "Incorrect Volume Size";

    CLOG(INFO, "LOGGER_SYS") << "N = " << N;

    Volume refA, refB;

    CLOG(INFO, "LOGGER_SYS") << "Reading Volumes";

    refAImf.readVolume(refA);
    refBImf.readVolume(refB);

    CLOG(INFO, "LOGGER_SYS") << "Performing Fourier Transform";
    
    FFT fft;
    fft.fw(refA);
    fft.fw(refB);

    CLOG(INFO, "LOGGER_SYS") << "Calculating FSC";

    vec fsc = vec(N / 2 - 1);
    FSC(fsc, refA, refB);

    double pixelSize = atof(argv[3]);

    for (int i = 1; i < N / 2 - 1; i++)
        printf("%05d   %10.6lf   %10.6lf\n",
               i,
               1.0 / resP2A(i, N, pixelSize),
               fsc(i));

    return 0;
}
