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

    Volume head(N, N, N, realSpace);
    //if (myid == messageid) {
    //    std::cout << "Define a head" << std::endl;
    //}
    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
            {
                if ((pow(x - N / 2, 2)
                   + pow(y - N / 2, 2)
                   + pow(z - N / 2, 2) < pow(N / 8, 2)) ||
                    (pow(x - N / 2, 2)
                   + pow(y - 3 * N / 8, 2)
                   + pow(z - 5 * N / 8, 2) < pow(N / 16, 2)) ||
                    (pow(x - N / 2, 2)
                   + pow(y - 5 * N / 8, 2) 
                   + pow(z - 5 * N / 8, 2) < pow(N / 16, 2)) ||
                    ((pow(x - N / 2, 2)
                    + pow(y - N / 2, 2) < pow(N / 16, 2)) &&
                     (z < 7 * N / 16) && (z > 5 * N / 16)))
                    head.setRL(1, x, y, z);
                else
                    head.setRL(0, x, y, z);
            }
    
    if (myid == server) {
        ImageFile imf;
        imf.readMetaData(head);
        imf.display();
        imf.writeImage("head.mrc", head);
    }
    FFT fft;
    fft.fw(head);

    /***
    VOLUME_FOR_EACH_PIXEL_FT(head)
        printf("%f %f\n", REAL(head.getFT(i, j, k)),
                          IMAG(head.getFT(i, j, k)));
    ***/

    Projector projector;
    projector.setProjectee(head);

    char name[256];
    int counter = 0;

    Image image(N, N, fourierSpace);
    // Image image(N, N, realSpace);
    
    Reconstructor reconstructor(N, N, N, N, N, 2, 1.9, 10);

    reconstructor.setCommRank(myid);
    reconstructor.setCommSize(numprocs);

    if (myid != server) {
        for (int k = M / numworkers * workerid; k < M / numworkers * (workerid + 1); k++)
        {    
            for (int j = 0; j < M / 2; j++)
            {
                for (int i = 0; i < M; i++)
                {
                    printf("%02d %02d %02d\n", i, j, k);
                    sprintf(name, "%02d%02d%02d.bmp", i, j, k);
                    Coordinate5D coord(2 * M_PI * i / M,
                                       2 * M_PI * j / M,
                                       2 * M_PI * k / M,
                                       0,
                                       0);
                    projector.project(image, coord);
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

#ifdef DEBUGAFTERINSERT
        reconstructor.display(TESTNODE, "testFWC-afterinsert");
#endif

        for (int i = 0; i < 3; i++) {
            reconstructor.allReduceW(workers);
            std::cout << "Round-" << i << ":       worker-" << workerid << "    :finised allreduce" << std::endl;
#ifdef DEBUGAFTERALLREDUCE
            char name[256];
            sprintf(name, "testFWC-afterallreduce%d", i);
            reconstructor.display(TESTNODE, name);
#endif
            std::cout << "checkC = " << reconstructor.checkC() << std::endl;
        }


    }

    reconstructor.reduceF(0, world);
#ifdef DEBUGAFTERREDUCEF
    reconstructor.display(server, "testFWC-afterreduceF");
#endif
 
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
