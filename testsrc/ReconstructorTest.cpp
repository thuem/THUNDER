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
#include "ImageFunctions.h"
#include "FFT.h"
#include "Mask.h"
#include "ImageFile.h"
#include "Timer.h"

#define N 256
#define M 32

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    MPI_Init(&argc, &argv);

    fftw_init_threads();

    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

    Image head(N, N, RL_SPACE);

    IMAGE_FOR_EACH_PIXEL_RL(head)
    {
        if ((NORM(i, j) < N / 8) ||
            (NORM(i - N / 8, j - N / 8) < N / 16) ||
            (NORM(i + N / 8, j - N / 8) < N / 16))
            head.setRL(1, i, j);
        else
            head.setRL(0, i, j);
    }

    if (commRank == MASTER_ID)
        head.saveRLToBMP("head.bmp");

    printf("head defined\n");

    Image padHead;

    //VOL_PAD_RL(padHead, head, 2);
    IMG_PAD_RL(padHead, head, 2);

    if (commRank == MASTER_ID)
        padHead.saveRLToBMP("padHead.bmp");

    normalise(padHead);

    if (commRank == MASTER_ID)
        padHead.saveRLToBMP("normalisedPadHead.bmp");

    /***
    std::cout << "Adding Noise" << std::endl;
    Volume noise(2 * N, 2 * N, 2 * N, RL_SPACE);
    gsl_rng* engine = get_random_engine();
    FOR_EACH_PIXEL_RL(noise)
        noise(i) = gsl_ran_gaussian(engine, 5);
    ADD_RL(padHead, noise);

    printf("padHead: mean = %f, stddev = %f, maxValue = %f\n",
           gsl_stats_mean(&padHead(0), 1, padHead.sizeRL()),
           gsl_stats_sd(&padHead(0), 1, padHead.sizeRL()),
           padHead(cblas_idamax(padHead.sizeRL(), &padHead(0), 1)));
    ***/

    FFT fft;
    fft.fw(padHead);
    printf("FFT Done\n");

    Projector projector;
    projector.setMode(MODE_2D);
    projector.setProjectee(padHead.copyImage());

    Image projectee = projector.projectee2D().copyImage();

    fft.bw(projectee);

    projectee.saveRLToBMP("projectee.bmp");

    char name[256];

    Image image(N, N, FT_SPACE);

    Image ctf(N, N, FT_SPACE);

    SET_1_FT(ctf);

    Reconstructor reconstructor(MODE_2D, N, 2, NULL);

    reconstructor.setMPIEnv();

    if (commRank != MASTER_ID)
    {
        printf("Projection and Insertion\n");

        for (int k = M / (commSize - 1) * (commRank - 1);
                 k < M / (commSize - 1) * commRank;
                 k++)
        {
            SET_0_FT(image);

            printf("%02d\n", k);
            sprintf(name, "%02d.bmp", k);

            double phi = 2 * M_PI / M * k;

            mat22 rot;

            rotate2D(rot, phi);

            projector.project(image, rot);

            fft.bw(image);

            image.saveRLToBMP(name);

            fft.fw(image);

            reconstructor.insert(image, ctf, rot, vec2(0, 0), 1);
        }
    }

    Image result;

    if (commRank != MASTER_ID)
    {
        if (commRank == HEMI_A_LEAD)
            CLOG(INFO, "LOGGER_SYS") << "Start Reconstruction";

        reconstructor.reconstruct(result);

        if (commRank == HEMI_A_LEAD)
            CLOG(INFO, "LOGGER_SYS") << "End Reconstruction";

        if (commRank == HEMI_A_LEAD)
        {
            result.saveRLToBMP("result.bmp");
        }
    }

    MPI_Finalize();

    return 0;
}
