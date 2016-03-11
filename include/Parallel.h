/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef PARALLEL_H
#define PARALLEL_H

#include <mpi.h>

#define SERVER_ID 0

class Parallel
{
    protected:

        int _commSize;
        int _commRank;

        /***
        MPI_Comm _world = MPI_COMM_WORLD;

        MPI_Comm _partWorld;
        ***/
        
        MPI_Comm _hemi;

    public:

        Parallel();

        ~Parallel();

        void setMPIEnv(int argc,
                       char* argv[]);

        void setMPIEnv(const int commSize,
                       const int commRank,
                       const MPI_Comm& hemi);

        bool isMaster() const;

        int commSize() const;

        void setCommSize(const int commSize);

        int commRank() const;

        void setCommRank(const int commRank);

        void setHemi(const MPI_Comm& hemi);

        /***
        Parallel(int argc,
                 char* argv[]);

        void init(int argc,
                  char* argv[]);
                  ***/
};

#endif // PARALLEL_H
