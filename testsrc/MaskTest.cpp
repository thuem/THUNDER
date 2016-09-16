/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include <omp.h>

#include <gsl/gsl_statistics.h>

#include "ImageFile.h"
#include "Mask.h"
#include "FFT.h"
#include "Filter.h"

#define PF 2

#define N 380

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    Volume ref;
    ImageFile imf("ref.mrc", "r");
    imf.readMetaData();
    imf.readVolume(ref);

    FFT fft;
    fft.fwMT(ref);

    lowPassFilter(ref, ref, 1.32 / 20, 2.0 / N);

    fft.bwMT(ref);

    Volume mask(N, N, N, RL_SPACE);

    //genMask(mask, ref, 0.1, 4);
    //genMask(mask, ref, 10, 3, 6, N * 0.5);
    //genMask(mask, ref, 10, 3, N * 0.5);
    //genMask(mask, ref, N * 0.5);
    //genMask(mask, ref, 10, 3, 2, N * 0.5);
    //genMask(mask, ref, 10, 1, 2, N * 0.5);
    genMask(mask, ref, GEN_MASK_EXT, GEN_MASK_EDGE_WIDTH, N * 0.5);

    ImageFile out;
    out.readMetaData(mask);
    out.writeVolume("mask.mrc", mask);
}
