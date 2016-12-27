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
//#define M 16



INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    /***
    loggerInit(argc, argv);

    MPI_Init(&argc, &argv);

    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

    // std::cout << "0: commRank = " << commRank << std::endl;
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

    if (commRank == MASTER_ID)
    {
        ImageFile imf;
        imf.readMetaData(head);
        imf.display();
        imf.writeVolume("head.mrc", head);
    }

    printf("head defined\n");

    Volume padHead;
    VOL_PAD_RL(padHead, head, 2);
    normalise(padHead);

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

    FFT fft;
    fft.fw(padHead);
    printf("FFT Done\n");

    Projector projector;
    projector.setProjectee(padHead.copyVolume());

    char name[256];

    Image image(N, N, FT_SPACE);
    // Image image(N, N, RL_SPACE);
    ***/

    Symmetry sym("C2");

    boost::container::vector< boost::movelib::unique_ptr<Reconstructor> > reco;

    reco.push_back(boost::movelib::unique_ptr<Reconstructor>(new Reconstructor()));
    reco[0]->init(N, 2, &sym);

    //Reconstructor reconstructor(N, 2, &sym);

    /***
    reconstructor.setMPIEnv();

    printf("Set Symmetry Done\n");

    try
    {
    if (commRank != MASTER_ID)
    {
        if (commRank == 1)
            timing();

        printf("Projection and Insertion\n");

        for (int k = M / (commSize - 1) * (commRank - 1);
                 k < M / (commSize - 1) * commRank;
                 k++)
            for (int j = 0; j < M / 2; j++)
                for (int i = 0; i < M / 2; i++)
                {
                    SET_0_FT(image);

                    printf("%02d %02d %02d\n", i, j, k);
                    sprintf(name, "%02d%02d%02d.bmp", i, j, k);
                    Coordinate5D coord(2 * M_PI * i / M,
                                       2 * M_PI * j / M,
                                       2 * M_PI * k / M,
                                       0,
                                       0);
                    projector.project(image, coord);

                    fft.bw(image);
                    
                    printf("image: mean = %f, stddev = %f, maxValue = %f\n",
                           gsl_stats_mean(&image(0), 1, image.sizeRL()),
                           gsl_stats_sd(&image(0), 1, image.sizeRL()),
                           image(cblas_idamax(image.sizeRL(), &image(0), 1)));

                    fft.fw(image);

                    reconstructor.insert(image, coord, 1);
                }
        }
        if (commRank == 1)
            timing();
    }
    }
    catch (Error& err)
    {
        std::cout << err << std::endl;
    }

    Volume result;
    if (commRank != MASTER_ID)
    {
        if (commRank == HEMI_A_LEAD)
            CLOG(INFO, "LOGGER_SYS") << "Start Reconstruction";

        reconstructor.reconstruct(result);

        if (commRank == HEMI_A_LEAD)
            CLOG(INFO, "LOGGER_SYS") << "End Reconstruction";

        printf("result: mean = %f, stddev = %f, maxValue = %f\n",
               gsl_stats_mean(&result(0), 1, result.sizeRL()),
               gsl_stats_sd(&result(0), 1, result.sizeRL()),
               result(cblas_idamax(result.sizeRL(), &result(0), 1)));

        if (commRank == 1)
        {
            ImageFile imf;
            imf.readMetaData(result);
            imf.writeVolume("result.mrc", result);
        }
    }

    if (commRank == 1)
        timing();

    if (commRank != MASTER_ID)
    {
        fft.fw(result);
        projector.setProjectee(result.copyVolume());
        for (int k = M / (commSize - 1) * (commRank - 1);
                 k < M / (commSize - 1) * commRank;
                 k++)
            for (int j = 0; j < M / 2; j++)
                for (int i = 0; i < M / 2; i++)
                {
                    SET_0_FT(image);

                    printf("%02d %02d %02d\n", i, j, k);
                    sprintf(name, "%02d%02d%02d.bmp", i, j, k);
                    Coordinate5D coord(2 * M_PI * i / M,
                                       2 * M_PI * j / M,
                                       2 * M_PI * k / M,
                                       0,
                                       0);
                    projector.project(image, coord);

                    fft.bw(image);

                    printf("image: mean = %f, stddev = %f, maxValue = %f\n",
                           gsl_stats_mean(&image(0), 1, image.sizeRL()),
                           gsl_stats_sd(&image(0), 1, image.sizeRL()),
                           image(cblas_idamax(image.sizeRL(), &image(0), 1)));

                    fft.fw(image);
                }
    }

    //MPI_Comm_free(&world);
    //MPI_Comm_free(&workers);
    //MPI_Group_free(&worker_group);
    //MPI_Group_free(&world_group);

    MPI_Finalize();
    return 0;
    ***/
}
