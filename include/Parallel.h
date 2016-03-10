
#ifndef PARALLEL_H
#define PARALLEL_H

#include <mpi.h>


class Parallel
{

    private:

        int _numProcs;
        int _myId;

        int _serverId;
        
        int _numWorkers = -1;
        int _workerId = -1;


        MPI_Comm _world = MPI_COMM_WORLD;

        MPI_Comm _partWorld;
        MPI_Comm _partWorldW;


    public:

        Parallel();

        ~Parallel();

        Parallel(int argc,
                 char* argv[]);

        void init(int argc,
                  char* argv[]);

};


#endif  //PARALLEL_H
