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

//#define N 256
#define N (576)
#define M 1

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    fftw_init_threads();

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

    ImageFile imf;

    imf.readMetaData(head);
    imf.writeVolume("head.mrc", head);

    FFT fft;

    CLOG(INFO, "LOGGER_SYS") << "Creating Plan";

    fft.fwCreatePlanMT(N, N, N);
    fft.bwCreatePlanMT(N, N, N);

    for (int k = 0; k < M; k++)
    {
        CLOG(INFO, "LOGGER_SYS") << "Executing Plan, Round " << k;

        fft.fwExecutePlanMT(head);

        /***
        FOR_EACH_PIXEL_FT(head)
            head[i] /= 2;
        ***/

        fft.bwExecutePlanMT(head);
    }

    CLOG(INFO, "LOGGER_SYS") << "Destroying Plan";

    fft.fwDestroyPlan();
    fft.bwDestroyPlan();

    imf.readMetaData(head);
    imf.writeVolume("head_2.mrc", head);

    fftw_cleanup_threads();

    return 0;
}
