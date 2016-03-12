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
#include "Mask.h"

#define N 16 
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
    
    // cout << "1: commRank = " << commRank << endl;
    if (commRank == MASTER_ID)
    {
        ImageFile imf;
        imf.readMetaData(head);
        imf.display();
        imf.writeVolume("head.mrc", head);
    }
    // cout << "2: commRank = " << commRank << endl;

    printf("head defined\n");

    Volume padHead;
    VOL_PAD_RL(padHead, head, 2);
    FFT fft;
    fft.fw(padHead);
    printf("FFT Done\n");
    // cout << "3: commRank = " << commRank << endl;

    Projector projector;
    projector.setProjectee(padHead);

    char name[256];
    int counter = 0;

    Image image(N, N, FT_SPACE);
    // Image image(N, N, RL_SPACE);
    
    Symmetry sym("C2");
   
    // cout << "4: commRank = " << commRank << endl;
    Reconstructor reconstructor(N, 2, &sym);
    reconstructor.setMPIEnv();

    printf("Set Symmetry Done\n");

    // cout << "5: commRank = " << commRank << endl;
    // cout << (commRank != MASTER_ID) << endl;

    if (commRank != MASTER_ID)
    {
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
                    printf("%02d %02d %02d\n", i, j, k);
                    sprintf(name, "%02d%02d%02d.bmp", i, j, k);
                    Coordinate5D coord(2 * M_PI * i / M,
                                       2 * M_PI * j / M,
                                       2 * M_PI * k / M,
                                       0,
                                       0);
                    projector.project(image, coord);
                    // C2C_RL(image, image, softMask(image, N / 4, 2));
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

                    // image.saveRLToBMP(name);
                    // image.saveFTToBMP(name, 0.1);    
                }

        for (int i = 0; i < 3; i++) {
            reconstructor.allReduceW();
            std::cout << "Round-" << i << ":       worker-" << commRank - 1 << "    :finised allreduce" << std::endl;
            std::cout << "checkC = " << reconstructor.checkC() << std::endl;
        }
    }

    /***
    reconstructor.allReduceF();
 
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
