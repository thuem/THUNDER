/*******************************************************************************
 * Author: Mingxu Hu, Bing Li
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "Macro.h"
#include "Typedef.h"
#include "Logging.h"
#include "Precision.h"

#include "ImageFile.h"
#include "Image.h"
#include "Volume.h"
#include "Spectrum.h"
#include "FFT.h"
#include "Mask.h"
#include "Filter.h"

/**
 * lower resolution limit for estimating B-factor in Angstrom
 */
#define B_FACTOR_EST_LOW_RES 10

class Postprocess
{
    private:

        int _size;

        RFLOAT _pixelSize;

        Volume _mapA;

        Volume _mapB;

        Volume _mapAMasked;

        Volume _mapBMasked;

        Volume _mapI;

        Volume _mapARFMask;

        Volume _mapBRFMask;

        Volume _mask;

        vec _FSCUnmask;

        vec _FSCMask;

        vec _FSCRFMask;

        vec _FSC;
        
        int _res;

    public:        

        Postprocess();

        Postprocess(const char mapAFilename[],
                    const char mapBFilename[],
                    const char maskFilename[],
                    const RFLOAT pixelSize);

        void run(const unsigned int nThread);

    private:

        /**
         * perform masking on reference A and reference B
         */
        void maskAB(const unsigned int nThread);

        void maskABRF(const unsigned int nThread);

        void randomPhaseAB(const int randomPhaseThres,
                           const unsigned int nThread);

        void mergeAB();

        int maxR();

        void saveFSC() const;
};

#endif // PREPROCESS_H
