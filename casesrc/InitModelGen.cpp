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

#define N 256

using namespace std;

int main(int argc, char* argv[])
{
    cout << "Defining Sphere" << endl;
    Volume sphere(N, N, N, RL_SPACE);
    VOLUME_FOR_EACH_PIXEL_RL(sphere)
    {
        if (NORM_3(i, j, k) < N / 8)
            sphere.setRL(1, i, j, k);
        else
            sphere.setRL(0, i, j, k);
    }
    normalise(sphere);

    ImageFile imf;
    imf.readMetaData(sphere);
    imf.writeVolume("sphere.mrc", sphere);

    return 0;
}
