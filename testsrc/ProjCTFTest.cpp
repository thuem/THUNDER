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
#include "CTF.h"

#define N 256
#define M 10
#define PF 2

#define PIXEL_SIZE 1.32
#define VOLTAGE 3e5
#define DEFOCUS 2e4
#define THETA 0
#define CS 0

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    FFT fft;

    std::cout << "Define a head." << std::endl;

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

    Volume padHead;
    VOL_PAD_RL(padHead, head, PF);

    fft.fw(padHead);

    Image ctf(N, N, FT_SPACE);

    CTF(ctf,
        PIXEL_SIZE,
        VOLTAGE,
        DEFOCUS,
        DEFOCUS,
        THETA,
        CS);

    Projector projector;
    projector.setPf(PF);
    projector.setProjectee(padHead.copyVolume());

    Image image(N, N, FT_SPACE);
    SET_0_FT(image);

    projector.project(image, 0, 0, 0);

    FOR_EACH_PIXEL_FT(image)
        image[i] *= gsl_pow_2(REAL(ctf[i]));

    fft.bw(image);

    image.saveRLToBMP("1.bmp");

    return 0;
}
