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

    CLOG(INFO, "LOGGER_SYS") << "Reading Map";

    ImageFile imf(argv[2], "rb");
    imf.readMetaData();

    Volume ref;
    imf.readVolume(ref);

    CLOG(INFO, "LOGGER_SYS") << "Removing Corners of the Map";

    softMask(ref, ref, atof(argv[3]), EDGE_WIDTH_RL, 0);

    CLOG(INFO, "LOGGER_SYS") << "Writing Map";

    imf.readMetaData(ref);

    imf.writeVolume(argv[1], ref, atof(argv[4]));
}
