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
        if ((abs(i) < N / 8) &&
            (abs(j) < N / 8) &&
            (abs(k) < N / 8))
            sphere.setRL(1, i, j, k);
        else
            sphere.setRL(0, i, j, k);
    }
    normalise(sphere);

    Volume padSphere;
    VOL_PAD_RL(padSphere, sphere, PF);

    ImageFile imf;
    imf.readMetaData(padSphere);
    imf.writeVolume("sphere.mrc", padSphere);

    return 0;
}
