/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "ImageFile.h"
#include "Volume.h"
#include "ImageFunctions.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, const char* argv[])
{
    loggerInit();

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

    Volume refA, refB;

    refAImf.readVolume(refA);
    refBImf.readVolume(refB);

    return 0;
}
