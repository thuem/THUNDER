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

    // REMOVE_NEG(_mapA);
    // REMOVE_NEG(_mapB);

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

    _mapAMasked.alloc(_size, _size, _size, RL_SPACE);
    _mapBMasked.alloc(_size, _size, _size, RL_SPACE);

    maskAB();

    imf.readMetaData(_mapAMasked);
    imf.writeVolume("Reference_A_Masked.mrc", _mapA);
    imf.readMetaData(_mapBMasked);
    imf.writeVolume("Reference_B_Masked.mrc", _mapB);

    CLOG(INFO, "LOGGER_SYS") << "Performing Fourier Transform";

    fft.fwMT(_mapA);
    fft.fwMT(_mapB);

    fft.fwMT(_mapAMasked);
    fft.fwMT(_mapBMasked);

    CLOG(INFO, "LOGGER_SYS") << "Determining FSCUnmask & FSCMask";

    _FSCUnmask.resize(maxR());

    _FSCMask.resize(maxR());

    FSC(_FSCUnmask, _mapA, _mapB);

    FSC(_FSCMask, _mapAMasked, _mapBMasked);

    int randomPhaseThres = resP(_FSCUnmask, 0.8, 1, 1, false);

    CLOG(INFO, "LOGGER_SYS") << "Performing Random Phase From "
                             << 1.0 / resP2A(randomPhaseThres,
                                             _size,
                                             _pixelSize);

    randomPhaseAB(randomPhaseThres);

    CLOG(INFO, "LOGGER_SYS") << "Determing FSCRFMask";

    fft.bwMT(_mapARFMask);
    fft.bwMT(_mapBRFMask);

    maskABRF();

    fft.fwMT(_mapARFMask);
    fft.fwMT(_mapBRFMask);

    _FSCRFMask.resize(maxR());

    FSC(_FSCRFMask, _mapARFMask, _mapBRFMask);

    CLOG(INFO, "LOGGER_SYS") << "Calculating True FSC";

    _FSC.resize(maxR());

    for (int i = 0; i < maxR(); i++)
    {
        if (i < randomPhaseThres + 2)
            _FSC(i) = _FSCMask(i);
        else
            _FSC(i) = (_FSCMask(i) - _FSCRFMask(i)) / (1 - _FSCRFMask(i));
    }

    CLOG(INFO, "LOGGER_SYS") << "Saving FSC";

    saveFSC();

    _res = resP(_FSC, 0.143, 1, 1, true);

    CLOG(INFO, "LOGGER_SYS") << "Resolution: "
                             << 1.0 / resP2A(_res,
                                             _size,
                                             _pixelSize);

    CLOG(INFO, "LOGGER_SYS") << "Merging Two References";
    
    mergeAB();

    fft.bw(_mapI);

    imf.readMetaData(_mapI);
    imf.writeVolume("Reference_Average.mrc", _mapI, _pixelSize);

    fft.fw(_mapI);

    CLOG(INFO, "LOGGER_SYS") << "Applying FSC Weighting";

    fscWeightingFilter(_mapI, _mapI, _FSC);

    CLOG(INFO, "LOGGER_SYS") << "Estimating B-Factor";

    double bFactor;
    
    bFactorEst(bFactor,
               _mapI,
               _res,
               AROUND(resA2P(1.0 / B_FACTOR_EST_LOW_RES, _size, _pixelSize)));

    //bFactor = -40;

    CLOG(INFO, "LOGGER_SYS") << "B-Factor : " << bFactor;

    CLOG(INFO, "LOGGER_SYS") << "Performing Sharpening";
    
    sharpen(_mapI,
            _mapI,
            (double)_res / _size,
            (double)EDGE_WIDTH_FT / _size,
            bFactor);

    //CLOG(INFO, "LOGGER_SYS") << "Compensating B-Factor Filtering";

    //bFactorFilter(_mapI, _mapI, COMPENSATE_B_FACTOR / gsl_pow_2(_pixelSize));

    CLOG(INFO, "LOGGER_SYS") << "Saving Result";

    fft.bw(_mapI);

    softMask(_mapI, _mapI, _mask, 0);

    //REMOVE_NEG(_mapI);

    imf.readMetaData(_mapI);
    imf.writeVolume("Reference_Sharp.mrc", _mapI, _pixelSize);
}

void Postprocess::maskAB()
{
    softMask(_mapAMasked, _mapA, _mask, 0);
    softMask(_mapBMasked, _mapB, _mask, 0);
}

void Postprocess::maskABRF()
{
    softMask(_mapARFMask, _mapARFMask, _mask, 0);
    softMask(_mapBRFMask, _mapBRFMask, _mask, 0);
}

void Postprocess::randomPhaseAB(const int randomPhaseThres)
{
    _mapARFMask.alloc(_size, _size, _size, FT_SPACE);
    _mapBRFMask.alloc(_size, _size, _size, FT_SPACE);

    randomPhase(_mapARFMask, _mapA, randomPhaseThres);
    randomPhase(_mapBRFMask, _mapB, randomPhaseThres);
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

void Postprocess::saveFSC() const
{
    FILE* file = fopen("Postprocess_FSC.txt", "w");

    for (int i = 1; i < _FSC.size(); i++)
        fprintf(file,
                "%05d   %10.6lf   %10.6lf\n",
                i,
                1.0 / resP2A(i, _size, _pixelSize),
                _FSC(i));

    fclose(file);
}
