
#ifndef PARALLEL_H
#define PARALLEL_H

#include <mpi.h>


class Parallel
{

    private:

        int _numProcs;
        int _myId;

        const int _serverId = 0;

        MPI_Comm _world = MPI_COMM_WORLD;

        MPI_Comm _partWorld;


    public:

        Parallel();

        ~Parallel();

        Parallel(int argc,
                 char* argv[]);

        void init(int argc,
                  char* argv[]);

};


#endif  //PARALLEL_H
