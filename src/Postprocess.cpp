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
                         const char mapBFilename[],
                         const double pixelSize)
{
    _pixelSize = pixelSize;

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
    FFT fft;

    ImageFile imf;

    CLOG(INFO, "LOGGER_SYS") << "Performing Fourier Transform";

    fft.fwMT(_mapA);
    fft.fwMT(_mapB);

    CLOG(INFO, "LOGGER_SYS") << "Determining FSC of Unmasked Half Maps";

    _fscU.resize(maxR());

    FSC(_fscU, _mapA, _mapB);

    CLOG(INFO, "LOGGER_SYS") << "Resolution of Unmasked Half Maps : "
                             << 1.0 / resP2A(resP(_fscU, 0.143),
                                             _size,
                                             _pixelSize);

    CLOG(INFO, "LOGGER_SYS") << "Averaging Half Maps";

    _mapI.alloc(_size, _size, _size, RL_SPACE);

    #pragma omp parallel for
    FOR_EACH_PIXEL_RL(_mapI)
        _mapI(i) = (_mapA(i) + _mapB(i)) / 2;

    CLOG(INFO, "LOGGER_SYS") << "Generating Mask";

    _mask.alloc(_size, _size, _size, RL_SPACE);

    genMask(_mask, _mapI, GEN_MASK_EXT, GEN_MASK_EDGE_WIDTH, _size * 0.5);

    CLOG(INFO, "LOGGER_SYS") << "Saving Mask";

    imf.readMetaData(_mask);
    imf.writeVolume("Postprocess_Mask", _mask);

    CLOG(INFO, "LOGGER_SYS") << "Determing Random Phase Resolution";

    int randomPhaseRes = resP(_fscU, RANDOM_PHASE_THRES);

    CLOG(INFO, "LOGGER_SYS") << "Performing Random Phase Above Frequency "
                             << 1.0 / resP2A(randomPhaseRes,
                                             _size,
                                             _pixelSize);
}

int Postprocess::maxR()
{
    return _size / 2 - 1;
}
