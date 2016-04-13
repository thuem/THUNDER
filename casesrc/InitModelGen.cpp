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
#define PF 1

using namespace std;

int main(int argc, char* argv[])
{
    cout << "Defining Sphere" << endl;
    Volume sphere(N, N, N, RL_SPACE);
    VOLUME_FOR_EACH_PIXEL_RL(sphere)
    {
        if ((NORM_3(i, j, k) < N / 8) ||
            (NORM_3(i - N / 8, j, k - N / 8) < N / 16) ||
            (NORM_3(i + N / 8, j, k - N / 8) < N / 16) ||
            ((NORM(i, j) < N / 16) &&
             (k + N / 16 < 0) &&
             (k + 3 * N / 16 > 0)))
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
