/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <fstream>

#include <json/json.h>

#include "FFT.h"
#include "ImageFile.h"
#include "Volume.h"

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    ImageFile imfSrc(argv[2], "rb");

    Volume src;

    imfSrc.readMetaData();
    imfSrc.readVolume(src);

    FFT fft;
    fft.fw(src);

    int size = atoi(argv[3]);

    Volume dst(size, size, size, FT_SPACE);

    if (src.nColRL() >= dst.nColRL())
    {
        #pragma omp parallel for
        VOLUME_FOR_EACH_PIXEL_FT(dst)
            dst.setFTHalf(src.getFTHalf(i, j, k), i, j, k);
    }
    else
    {
        #pragma omp parallel for
        SET_0_FT(dst);

        #pragma omp parallel for
        VOLUME_FOR_EACH_PIXEL_FT(src)
            dst.setFTHalf(src.getFTHalf(i, j, k), i, j, k);
    }

    fft.bw(dst);

    ImageFile imfDst;

    imfDst.readMetaData(dst);
    imfDst.writeVolume(argv[1], dst, atof(argv[4]));

    return 0;
}
