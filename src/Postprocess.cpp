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
                         const char maskFilename[],
                         const double pixelSize)
{
    _pixelSize = pixelSize;

    ImageFile imfA(mapAFilename, "rb");
    ImageFile imfB(mapBFilename, "rb");
    ImageFile imfM(maskFilename, "rb");

    imfA.readMetaData();
    imfB.readMetaData();
    imfM.readMetaData();

    CLOG(INFO, "LOGGER_SYS") << "Reading Two Half Maps";

    imfA.readVolume(_mapA);
    imfB.readVolume(_mapB);
    imfM.readVolume(_mask);

    _size = _mapA.nColRL();

    if ((_size != _mapA.nRowRL()) ||
        (_size != _mapA.nSlcRL()) ||
        (_size != _mapB.nColRL()) ||
        (_size != _mapB.nRowRL()) ||
        (_size != _mapB.nSlcRL()) ||
        (_size != _mask.nColRL()) ||
        (_size != _mask.nRowRL()) ||
        (_size != _mask.nSlcRL()))
        CLOG(FATAL, "LOGGER_SYS") << "Invalid Input Half Maps in Postprocessing";
}

void Postprocess::run()
{
    FFT fft;

    ImageFile imf;

    CLOG(INFO, "LOGGER_SYS") << "Masking Reference A and B";

    maskAB();

    imf.readMetaData(_mapA);
    imf.writeVolume("Reference_A_Masked.mrc", _mapA);
    imf.readMetaData(_mapB);
    imf.writeVolume("Reference_B_Masked.mrc", _mapB);

    CLOG(INFO, "LOGGER_SYS") << "Performing Fourier Transform";

    fft.fwMT(_mapA);
    fft.fwMT(_mapB);

    CLOG(INFO, "LOGGER_SYS") << "Determining FSC";

    _FSC.resize(maxR());

    FSC(_FSC, _mapA, _mapB);

    _res = resP(_FSC, 0.143);

    CLOG(INFO, "LOGGER_SYS") << "Resolution: "
                             << 1.0 / resP2A(_res,
                                             _size,
                                             _pixelSize);

    CLOG(INFO, "LOGGER_SYS") << "Merging Two References";
    
    mergeAB();

    CLOG(INFO, "LOGGER_SYS") << "Applying FSC Weighting";

    fscWeightingFilter(_mapI, _mapI, _FSC);

    CLOG(INFO, "LOGGER_SYS") << "Estimating B-Factor";

    double bFactor;
    bFactorEst(bFactor,
               _mapI,
               _res,
               AROUND(resA2P(1.0 / A_B_AVERAGE_THRES, _size, _pixelSize)));

    CLOG(INFO, "LOGGER_SYS") << "B-Factor : " << bFactor;

    CLOG(INFO, "LOGGER_SYS") << "Performing Sharpening";
    
    sharpen(_mapI,
            _mapI,
            (double)_res / _size,
            (double)EDGE_WIDTH_FT / _size,
            bFactor);

    CLOG(INFO, "LOGGER_SYS") << "Saving Result";

    fft.bw(_mapI);
    REMOVE_NEG(_mapI);

    imf.readMetaData(_mapI);
    imf.writeVolume("Reference_Sharp.mrc", _mapI, _pixelSize);
    //imf.writeVolume("Reference_Sharp.mrc", _mapI, _pixelSize);

    /***
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

    fft.fw(_mapI);

    Volume lowPassMapI(_size, _size, _size, FT_SPACE);
    lowPassFilter(lowPassMapI,
                  _mapI,
                  _pixelSize / GEN_MASK_RES,
                  (double)EDGE_WIDTH_FT / _size);

    fft.bw(lowPassMapI);

    lowPassMapI.clearFT();

    autoMask(_mask, lowPassMapI, GEN_MASK_EXT, GEN_MASK_EDGE_WIDTH, _size * 0.5);

    lowPassMapI.clearRL();

    CLOG(INFO, "LOGGER_SYS") << "Saving Mask";

    imf.readMetaData(_mask);
    imf.writeVolume("Postprocess_Mask.mrc", _mask);

    CLOG(INFO, "LOGGER_SYS") << "Performing Mask on Half Maps";

    _mapAMask.alloc(_size, _size, _size, RL_SPACE);
    _mapBMask.alloc(_size, _size, _size, RL_SPACE);

    #pragma omp parallel for
    FOR_EACH_PIXEL_RL(_mask)
    {
        _mapAMask(i) = _mapA(i) * _mask(i);
        _mapBMask(i) = _mapB(i) * _mask(i);
    }

    CLOG(INFO, "LOGGER_SYS") << "Determining FSC of Masked Half Maps";

    fft.fwMT(_mapAMask);
    fft.fwMT(_mapBMask);

    _fscM.resize(maxR());

    FSC(_fscM, _mapAMask, _mapBMask);

    CLOG(INFO, "LOGGER_SYS") << "Determing Random Phase Resolution";

    int randomPhaseRes = resP(_fscU, RANDOM_PHASE_THRES);

    CLOG(INFO, "LOGGER_SYS") << "Performing Random Phase Above Frequency "
                             << 1.0 / resP2A(randomPhaseRes,
                                             _size,
                                             _pixelSize);

    _mapARFMask.alloc(_size, _size, _size, FT_SPACE);
    _mapBRFMask.alloc(_size, _size, _size, FT_SPACE);

    randomPhase(_mapARFMask, _mapA, randomPhaseRes);
    randomPhase(_mapBRFMask, _mapB, randomPhaseRes);

    fft.bwMT(_mapARFMask);
    fft.bwMT(_mapBRFMask);

    CLOG(INFO, "LOGGER_SYS") << "Performing Mask on Random Phase Maps";

    #pragma omp parallel for
    FOR_EACH_PIXEL_RL(_mask)
    {
        _mapARFMask(i) *= _mask(i);
        _mapBRFMask(i) *= _mask(i);
    }

    CLOG(INFO, "LOGGER_SYS") << "Determining FSC of Random Phase Masked Half Maps";

    fft.fwMT(_mapARFMask);
    fft.fwMT(_mapBRFMask);

    _fscR.resize(maxR());

    FSC(_fscR, _mapARFMask, _mapBRFMask);

    CLOG(INFO, "LOGGER_SYS") << "Calculating True FSC";

    _fscT = (_fscM.array() - _fscR.array()) / (1 - _fscR.array());

    FILE* file = fopen("FSC.txt", "w");
    for (int i = 1; i < maxR(); i++)
        fprintf(file,
                "%04d %06f %06f %06f\n",
                i,
                _fscM(i),
                _fscR(i),
                _fscT(i));
    fclose(file);

    CLOG(INFO, "LOGGER_SYS") << "True Resolution : "
                             << 1.0 / resP2A(resP(_fscT,
                                                  0.143,
                                                  1,
                                                  randomPhaseRes + 2),
                                             _size,
                                             _pixelSize);
    ***/
}

void Postprocess::maskAB()
{
    softMask(_mapA, _mapA, _mask, 0);
    softMask(_mapB, _mapB, _mask, 0);
}

void Postprocess::mergeAB()
{
    _mapI.alloc(_size, _size, _size, FT_SPACE);

    FOR_EACH_PIXEL_FT(_mapI)
        _mapI[i] = (_mapA[i] + _mapB[i]) / 2;
}

int Postprocess::maxR()
{
    return _size / 2 - 1;
}
