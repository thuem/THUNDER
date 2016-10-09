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

#include "ImageFile.h"
#include "Image.h"
#include "Volume.h"
#include "Spectrum.h"
#include "FFT.h"
#include "Mask.h"

#define RANDOM_PHASE_THRES 0.8

using namespace std;

class Postprocess
{
    private:

        int _size;

        double _pixelSize;

        Volume _mapA;

        Volume _mapB;

        Volume _mapI;

        Volume _mapAMask;

        Volume _mapBMask;

        Volume _mapARFMask;

        Volume _mapBRFMask;

        Volume _mask;

        /**
         * FSC of two unmasked half maps
         */
        vec _fscU;

        /**
         * FSC of two masked half maps
         */
        vec _fscM;

        /**
         * FSC of two random-phase masked half maps
         */
        vec _fscR;

        /**
         * true FSC
         */
        vec _fscT;

    public:        

        Postprocess();

        Postprocess(const char mapAFilename[],
                    const char mapBFilename[],
                    const double pixelSize);

        void run();

    private:

        int maxR();
};

#endif // PREPROCESS_H
