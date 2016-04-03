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

#define N 128
#define M 16

using namespace std;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
      
    // cout << "0: commRank = " << commRank << endl;
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

    cout << "Adding Noise" << endl;
    Volume noise(2 * N, 2 * N, 2 * N, RL_SPACE);
    FOR_EACH_PIXEL_RL(noise)
        noise(i) = gsl_ran_gaussian(RANDR, 5);
    ADD_RL(padHead, noise);

    printf("padHead: mean = %f, stddev = %f, maxValue = %f\n",
           gsl_stats_mean(&padHead(0), 1, padHead.sizeRL()),
           gsl_stats_sd(&padHead(0), 1, padHead.sizeRL()),
           padHead(cblas_idamax(padHead.sizeRL(), &padHead(0), 1)));

    FFT fft;
    fft.fw(padHead);
    printf("FFT Done\n");

    Projector projector;
    projector.setProjectee(padHead);

    char name[256];
    int counter = 0;

    Image image(N, N, FT_SPACE);
    // Image image(N, N, RL_SPACE);
    
    Symmetry sym("C2");
   
    Reconstructor reconstructor(N, 2, &sym);
    reconstructor.setMPIEnv();

    printf("Set Symmetry Done\n");

    try
    {
    if (commRank != MASTER_ID)
    {
        if (commRank == 1)
            timing();

        printf("Projection and Insertion\n");
        /***
        for (int k = M / 2 / numworkers * workerid;
                 k < M / 2 / numworkers * (workerid + 1);
                 k++)
                 ***/

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

                    /***
                    FFT fft;
                    fft.fw(image);
                    fft.bw(image);
                    ***/
                    /***
                    R2R_FT(image, sin(2));
                    ***/
                    /***
                    R2R_FT(image, image, projector.project(image,
                                                    2 * M_PI * i / M,
                                                    M_PI * j / M,
                                                    2 * M_PI * k / M));
                                                    ***/

                    fft.bw(image);
                    image.saveRLToBMP(name);
                    fft.fw(image);
                    // image.saveFTToBMP(name, 0.1);    
                }

        /***
        for (int i = 0; i < 3; i++) {
            reconstructor.allReduceW();
            std::cout << "Round-" << i << ":       worker-" << commRank - 1 << "    :finised allreduce" << std::endl;
            std::cout << "checkC = " << reconstructor.checkC() << std::endl;
        }
        ***/
        if (commRank == 1)
            timing();
    }
    }
    catch (Error& err)
    {
        cout << err << endl;
    }

    /***
    reconstructor.allReduceF();
    ***/
 
    Volume result;
    reconstructor.reconstruct(result);

    if (commRank == 1)
    {
        ImageFile imf;
        imf.readMetaData(result);
        imf.writeVolume("result.mrc", result);
    }

    if (commRank == 1)
        timing();

    /***
    if (commRank == 1) {
    // if (myid == 1) {
        std::cout << "server-" << commRank << ":          finised reduce" << std::endl;
        reconstructor.constructor("result.mrc");
        std::cout << "output success!" << std::endl;
    }
    ***/

    //MPI_Comm_free(&world);
    //MPI_Comm_free(&workers);
    //MPI_Group_free(&worker_group);
    //MPI_Group_free(&world_group);

    MPI_Finalize();
    return 0;
}
