/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include <omp_compat.h>

#include <gsl/gsl_statistics.h>

#include "ImageFile.h"
#include "Mask.h"
#include "FFT.h"
#include "Filter.h"

#define PF 2

#define N 380

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    Volume ref;
    ImageFile imfR(argv[1], "rb");
    imfR.readMetaData();
    imfR.readVolume(ref);

    Volume mask;
    ImageFile imfM(argv[2], "rb");
    imfM.readMetaData();
    imfM.readVolume(mask);

    FOR_EACH_PIXEL_RL(ref)
        ref(i) *= mask(i);

    imfR.readMetaData(ref);
    imfR.writeVolume("masked_ref.mrc", ref);

    /***
    FFT fft;
    fft.fwMT(ref);

    lowPassFilter(ref, ref, 1.32 / 30, 2.0 / N);

    fft.bwMT(ref);

    Volume mask(N, N, N, RL_SPACE);

    autoMask(mask, ref, GEN_MASK_EXT, GEN_MASK_EDGE_WIDTH, N * 0.5);

    ImageFile out;
    out.readMetaData(mask);
    out.writeVolume("mask.mrc", mask);
    ***/
}
