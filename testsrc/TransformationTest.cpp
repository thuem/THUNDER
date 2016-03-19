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

    VOLUME_FOR_EACH_PIXEL_RL(head)
    {
        if ((NORM_3(i, j, k) < N / 8) ||
            (NORM_3(i - N / 8, j, k - N / 8) < N / 16) ||
            (NORM_3(i + N / 8, j, k - N / 8) < N / 16) ||
            ((NORM(i, j) < N / 16) &&
             (k + N / 16 < 0) &&
             (k + 3 * N / 16 > 0)))
            head.setRL(1, i, j, k);
        else
            head.setRL(0, i, j, k);
    }
    
    ImageFile imf;
    imf.readMetaData(head);
    imf.writeVolume("head.mrc", head);

    Volume vol(N, N, N, RL_SPACE);
    VOL_TRANSFORM_RL(vol, head, mat, rotate3D(mat, M_PI / 2, 'Z'), N / 2 - 1);
    imf.readMetaData(vol);
    imf.writeVolume("rotate1.mrc", vol);

    Volume symVol;
    Symmetry sym("C4");
    // symmetryRL(symVol, head, sym, N / 2 - 1);
    SYMMETRIZE_RL(symVol, head, sym, N / 2 - 1);
    imf.readMetaData(symVol);
    imf.writeVolume("symVol.mrc", symVol);

    return 0;
}
