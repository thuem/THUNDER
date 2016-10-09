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
    CLOG(INFO, "LOGGER_SYS") << "Determining FSC of Unmasked Half Maps";

    _fscU.resize(maxR());

    FSC(_fscU, _mapA, _mapB);

    CLOG(INFO, "LOGGER_SYS") << "Resolution of Unmasked Half Maps : "
                             << 1.0;
}

int Postprocess::maxR()
{
    return _size / 2 - 1;
}
