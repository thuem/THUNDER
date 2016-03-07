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
#include "FFT.h"

#include "Transformation.h"

#define N 256
#define M 8

int main(int argc, const char* argv[])
{
    std::cout << "Define a head." << std::endl;

    Volume head(N, N, N, RL_SPACE);
    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
            {
                if ((pow(x - N / 2, 2)
                   + pow(y - N / 2, 2)
                   + pow(z - N / 2, 2) < pow(N / 8, 2)) ||
                    (pow(x - N / 2, 2)
                   + pow(y - 3 * N / 8, 2)
                   + pow(z - 5 * N / 8, 2) < pow(N / 16, 2)) ||
                    (pow(x - N / 2, 2)
                   + pow(y - 5 * N / 8, 2) 
                   + pow(z - 5 * N / 8, 2) < pow(N / 16, 2)) ||
                    ((pow(x - N / 2, 2)
                    + pow(y - N / 2, 2) < pow(N / 16, 2)) &&
                     (z < 7 * N / 16) && (z > 5 * N / 16)))
                    head.setRL(1, x - N / 2, y - N / 2, z - N / 2);
                else
                    head.setRL(0, x - N / 2, y - N / 2, z - N / 2);
            }
    
    ImageFile imf;
    imf.readMetaData(head);
    imf.writeVolume("head.mrc", head);

    Volume vol(N, N, N, RL_SPACE);
    VOL_TRANSFORM_RL(vol, head, mat, rotate3D(mat, M_PI / 8, 'Y'), N / 2 - 1);
    imf.readMetaData(vol);
    imf.writeVolume("rotate1.mrc", vol);

    Volume symVol;
    Symmetry sym("C2");
    // symmetryRL(symVol, head, sym, N / 2 - 1);
    SYMMETRIZE_RL(symVol, head, sym, N / 2 - 1);
    imf.readMetaData(symVol);
    imf.writeVolume("symVol.mrc", symVol);

    return 0;
}
