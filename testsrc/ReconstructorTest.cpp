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

#define N 32
#define M 16

#define DEBUGAFTERINSERT
#define DEBUGAFTERALLREDUCE
#define DEBUGAFTERREDUCEF


#ifdef DEBUGAFTERINSERT
    #define TESTNODE 1
#endif

#ifndef DEBUGAFTERINSERT
    #ifdef DEBUGAFTERALLREDUCE
        #define TESTNODE 1
    #endif
#endif



int main(int argc, char* argv[])
{

    int messageid = 2;
    int numprocs, myid, server, serverid[1];
    int numworkers, workerid;
    
    MPI_Comm world, workers;
    MPI_Group world_group, worker_group;

    MPI_Init(&argc, &argv);
    world = MPI_COMM_WORLD;
    MPI_Comm_size(world, &numprocs);
    MPI_Comm_rank(world, &myid);
      
    server = 0;
    MPI_Comm_group(world, &world_group);
    serverid[0] = server;
    MPI_Group_excl(world_group, 1, serverid, &worker_group);
    MPI_Comm_create(world, worker_group, &workers);

    if (myid == server) {
        std::cout << "1-Initial OK! , commited by server:" << server << std::endl;
    } else {
        MPI_Comm_rank(workers, &workerid);
        MPI_Comm_size(workers, &numworkers);
        std::cout << "1-Initial OK! , commited by worker:" << workerid << std::endl;
    }
    //if (myid == numprocs - 1) {
    //    std::cout << "2-numprocs: " << numprocs << std::endl
    //              << "2-myid: " << myid << std::endl
    //              << "2-server is node " << server << std::endl
    //              << "2-numworkers: " << numworkers << std::endl
    //              << "2-workerid: " << workerid << std::endl;
    //}

    Volume head(N, N, N, RL_SPACE);
    //if (myid == messageid) {
    //    std::cout << "Define a head" << std::endl;
    //}
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
    
    if (myid == server) {
        ImageFile imf;
        imf.readMetaData(head);
        imf.display();
        imf.writeVolume("head.mrc", head);
    }

    Volume padHead;
    VOL_PAD_RL(padHead, head, 2);
    FFT fft;
    fft.fw(padHead);

    /***
    VOLUME_FOR_EACH_PIXEL_FT(head)
        printf("%f %f\n", REAL(head.getFT(i, j, k)),
                          IMAG(head.getFT(i, j, k)));
    ***/

    Projector projector;
    projector.setProjectee(padHead);

    char name[256];
    int counter = 0;

    Image image(N, N, FT_SPACE);
    // Image image(N, N, RL_SPACE);
    
    Symmetry sym("C2");
   
    Reconstructor reconstructor(N, 2, &sym);

    reconstructor.setCommRank(myid);
    reconstructor.setCommSize(numprocs);

    if (myid != server) {
        for (int k = M / 2 / numworkers * workerid;
                 k < M / 2 / numworkers * (workerid + 1);
                 k++)
        {    
            for (int j = 0; j < M / 2; j++)
            {
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
                    reconstructor.insert(image, coord, 1, 1);
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
            }
        }

        for (int i = 0; i < 3; i++) {
            reconstructor.allReduceW(workers);
            std::cout << "Round-" << i << ":       worker-" << workerid << "    :finised allreduce" << std::endl;
            std::cout << "checkC = " << reconstructor.checkC() << std::endl;
        }


    }

    reconstructor.allReduceF(world);
 
    if (myid == server) {
    // if (myid == 1) {
        std::cout << "server-" << myid << ":          finised reduce" << std::endl;
        reconstructor.constructor("result.mrc");
        std::cout << "output success!" << std::endl;
    }

    //MPI_Comm_free(&world);
    //MPI_Comm_free(&workers);
    //MPI_Group_free(&worker_group);
    //MPI_Group_free(&world_group);

    MPI_Finalize();
    return 0;
}
