/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Projector.h"
#include "Reconstructor.h"
#include "FFT.h"
#include "ImageFile.h"
#include "Particle.h"

#define N 128
#define M 10000
#define MAX_X 30
#define MAX_Y 30

using namespace std;

int main(int argc, char* argv[])
{
    cout << "Defining Head" << endl;
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

    cout << "Padding Head" << endl;
    Volume padHead;
    VOL_PAD_RL(padHead, head, 2);

    cout << "Fourier Transforming Head" << endl;
    FFT fft;
    fft.fw(padHead);

    cout << "Setting Projectee" << endl;
    Projector projector;
    projector.setProjectee(padHead);

    char name[256];

    // Image image(N, N, FT_SPACE);
    Image image(N, N, RL_SPACE);

    cout << "Initialising Random Sampling Points" << endl;
    Particle par(M, MAX_X, MAX_Y);

    Coordinate5D coord;
    for (int i = 0; i < M; i++)
    {
        sprintf(name, "%06d.bmp", i);
        printf("%s\n", name);

        par.coord(coord, i);
        R2R_FT(image, image, projector.project(image, coord));
        image.saveRLToBMP(name);
    }
    
    return 0;
}
