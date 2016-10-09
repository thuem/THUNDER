/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Postprocess.h"

Postprocess::Postprocess() {}

Postprocess::Postprocess(const char mapAFilename[],
                         const char mapBFilename[])
{
    ImageFile imfA(mapAFilename, "rb");
    ImageFile imfB(mapBFilename, "rb");

    imfA.readMetaData();
    imfB.readMetaData();

    CLOG(INFO, "LOGGER_SYS") << "Reading Two Half Maps";

    imfA.readVolume(_mapA);
    imfB.readVolume(_mapB);

    _size = _mapA.nColRL();

    if ((_size != _mapA.nRowRL()) ||
        (_size != _mapA.nSlcRL()) ||
        (_size != _mapB.nColRL()) ||
        (_size != _mapB.nRowRL()) ||
        (_size != _mapB.nSlcRL()))
        CLOG(FATAL, "LOGGER_SYS") << "Invalid Input Half Maps in Postprocessing";
}

void Postprocess::run()
{
}
