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
#include "Mask.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    ImageFile imf(argv[2], "rb");
    imf.readMetaData();

    Volume ref;
    imf.readVolume(ref);

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(ref)
        if (QUAD_3(i, j, k) >= gsl_pow_2(ref.nColRL() / 2))
            ref.setRL(0, i, j, k);

    Volume mask(ref.nColRL(),
                ref.nRowRL(),
                ref.nSlcRL(),
                RL_SPACE);

    genMask(mask,
            ref,
            atof(argv[3]),
            atof(argv[4]),
            atof(argv[5]));

    imf.readMetaData(mask);

    imf.writeVolume(argv[1], mask, atof(argv[6]));
}
