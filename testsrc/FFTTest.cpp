/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include <mpi.h>

#include "FFT.h"
#include "ImageFile.h"

#define N 760
#define M 10
#define PF 4

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    MPI_Init(&argc, &argv);

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

    /***
    ImageFile imf;

    imf.readMetaData(head);
    imf.writeVolume("head.mrc", head);
    ***/

    FFT fft;
    fft.fwMT(head);
    fft.bwMT(head);

    /***
    imf.readMetaData(head);
    imf.writeVolume("head_2.mrc", head);
    ***/

    MPI_Finalize();
}
