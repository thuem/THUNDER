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

    FFT fft;
    fft.fw(head);

    Projector projector;
    projector.setMode(MODE_2D);
    projector.setProjectee(head.copyImage());

    Image projectee = projector.projectee2D().copyImage();

    fft.bw(projectee);

    projectee.saveRLToBMP("projectee.bmp");

    char name[256];

    Image image(N, N, FT_SPACE);

    Image ctf(N, N, FT_SPACE);

    SET_1_FT(ctf);

    Reconstructor reconstructor(MODE_2D, N, 2, NULL);

    reconstructor.setMPIEnv();

    reconstructor.setMaxRadius(13);

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

            reconstructor.insert(image, ctf, rot, 1);
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
