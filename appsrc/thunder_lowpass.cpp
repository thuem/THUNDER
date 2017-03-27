/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <fstream>

#include "ImageFile.h"
#include "Volume.h"
#include "Filter.h"
#include "FFT.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    fftw_init_threads();

    ImageFile imf(argv[2], "rb");
    imf.readMetaData();

    Volume ref;
    imf.readVolume(ref);

    FFT fft;
    fft.fw(ref);

    lowPassFilter(ref,
                  ref,
                  atof(argv[4]) / atof(argv[3]),
                  2.0 / ref.nColRL());

    fft.bwMT(ref);

    imf.readMetaData(ref);

    imf.writeVolume(argv[1], ref);

    fftw_cleanup_threads();

    return 0;
}
