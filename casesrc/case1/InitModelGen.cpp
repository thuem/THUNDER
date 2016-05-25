/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "ImageFile.h"
#include "Volume.h"

#define N 128
#define PF 2

using namespace std;

int main(int argc, char* argv[])
{
    cout << "Defining Sphere" << endl;
    Volume sphere(N, N, N, RL_SPACE);
    VOLUME_FOR_EACH_PIXEL_RL(sphere)
    {
        /***
        double ii = i * 0.8;
        double jj = j * 0.8;
        double kk = k * 0.8;
        if (NORM_3(ii, jj, kk) < N / 8)
            sphere.setRL(1, i, j, k);
        else
            sphere.setRL(0, i, j, k);
        ***/
        if ((NORM_3(ii, jj, kk) < N / 8) ||
            (NORM_3(ii - N / 8, jj, kk - N / 8) < N / 16) ||
            (NORM_3(ii + N / 8, jj, kk - N / 8) < N / 16) ||
            ((NORM(ii, jj) < N / 16) &&
             (kk + N / 16 < 0) &&
             (kk + 3 * N / 16 > 0)))
            sphere.setRL(1, i, j, k);
        else
            sphere.setRL(0, i, j, k);
    }

    Volume padSphere;
    VOL_PAD_RL(padSphere, sphere, PF);
    normalise(padSphere);

    ImageFile imf;
    imf.readMetaData(padSphere);
    imf.writeVolume("sphere.mrc", padSphere);

    return 0;
}
