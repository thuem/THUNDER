/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include <gsl/gsl_randist.h>

#include "Projector.h"
#include "ImageFile.h"
#include "FFT.h"
#include "Random.h"
#include "Transformation.h"

#define N 128
#define M 10
#define PF 2

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    ImageFile imf(argv[1], "r");

    imf.readMetaData();

    Volume ref;

    imf.readVolume(ref);

    FFT fft;

    fft.fwMT(ref);

    Projector projector;

    projector.setProjectee(ref.copyVolume());

    mat33 rot;

    Image img(ref.nColRL(), ref.nColRL(), FT_SPACE);

    char filename[FILE_NAME_LENGTH];

    for (int l = 0; l < 10; l++)
    {
        SET_0_FT(img);

        randRotate3D(rot);

        projector.projectMT(img, rot);

        fft.bw(img);

        sprintf(filename, "Image_%d.bmp", l);

        img.saveRLToBMP(filename);

        fft.fw(img);
    }

    /***
    Volume head(N, N, N, RL_SPACE);

    VOLUME_FOR_EACH_PIXEL_RL(head)
    {
        double ii = i * 0.8;
        double jj = j * 0.8;
        double kk = k * 0.8;
        if ((NORM_3(ii, jj, kk) < N / 8) ||
            (NORM_3(ii - N / 8, jj, kk - N / 8) < N / 16) ||
            (NORM_3(ii - N / 8, jj - N / 8, kk - N / 8) < N / 16) ||
            (NORM_3(ii + N / 8, jj, kk - N / 8) < N / 16) ||
            (NORM_3(ii + N / 8, jj + N / 8, kk - N / 8) < N / 16) ||
            ((NORM(ii, jj) < N / 16) &&
             (kk + N / 16 < 0) &&
             (kk + 3 * N / 16 > 0)))
            head.setRL(1, i, j, k);
        else
            head.setRL(0, i, j, k);
    }
    ***/
}
