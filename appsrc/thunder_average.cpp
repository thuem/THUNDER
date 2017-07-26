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

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    CLOG(INFO, "LOGGER_SYS") << "Reading Map";

    ImageFile imfA(argv[2], "rb");
    imfA.readMetaData();

    Volume refA;
    imfA.readVolume(refA);

    ImageFile imfB(argv[3], "rb");
    imfB.readMetaData();

    Volume refB;
    imfB.readVolume(refB);

    FOR_EACH_PIXEL_RL(refA)
    {
        refA(i) += refB(i);
        refA(i) /= 2;
    }

    ImageFile imf;
    imf.readMetaData(refA);
    imf.writeVolume(argv[1], refA, atof(argv[4]));

    return 0;
}
