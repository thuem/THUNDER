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

    ImageFile imf;

    //ImageFile imf(argv[2], "rb");
    //imf.readMetaData();

    /***
    CLOG(INFO, "LOGGER_SYS") << "Reading Map";

    Volume ref;
    imf.readVolume(ref);

    CLOG(INFO, "LOGGER_SYS") << "Removing Corners of the Map";

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(ref)
        if (QUAD_3(i, j, k) >= TSGSL_pow_2(ref.nColRL() / 2))
            ref.setRL(0, i, j, k);
    ***/

    CLOG(INFO, "LOGGER_SYS") << "Generating Mask";

    Volume mask(atoi(argv[5]),
                atoi(argv[5]),
                atoi(argv[5]),
                RL_SPACE);

    RFLOAT ew = atof(argv[4]);

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(mask)
    {
        RFLOAT d = NORM_3(i, j, k);

        RFLOAT inner = atof(argv[2]) / atof(argv[6]);
        RFLOAT outer = atof(argv[3]) / atof(argv[6]);

        if (d < inner - ew)
        {
            mask.setRL(0, i, j, k);
        }
        else if (d < inner)
        {
            mask.setRL(cos((d - inner) / ew) + 0.5, i, j, k);
        }
        else if (d < outer)
        {
            mask.setRL(1, i, j, k);
        }
        else if (d < outer + ew)
        {
            mask.setRL(cos((d - outer) / ew) + 0.5, i, j, k);
        }
        else
        {
            mask.setRL(0, i, j, k);
        }
    }

    //softEdge(mask, atof(argv[5]));

    /***
    genMask(mask,
            ref,
            atof(argv[3]),
            atof(argv[4]),
            atof(argv[5]));
    ***/

    CLOG(INFO, "LOGGER_SYS") << "Writing Mask";

    imf.readMetaData(mask);

    imf.writeVolume(argv[1], mask, atof(argv[6]));
}
