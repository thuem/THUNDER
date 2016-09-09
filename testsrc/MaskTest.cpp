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

    Volume mask(N, N, N, RL_SPACE);

    genMask(mask, ref, 0.02, 4);

    ImageFile out;
    out.readMetaData(mask);
    out.writeVolume("mask.mrc", mask);
}
