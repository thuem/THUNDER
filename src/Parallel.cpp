
#include "Parallel.h"

Parallel::Parallel() {}

Parallel::~Parallel() {}

Parallel::Parallel(int argc,
                   char* argv[])
{
    init(argc, argv);
}


void Parallel::init(int argc,
                    char* argv[])
{
    MPI_Group worldGroup;
    MPI_Group AGroup, BGroup;

    MPI_Comm A, B;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(_world, &_numProcs);
    MPI_Comm_rank(_world, &_myId);

    int half = (_numProcs - 1) / 2;
    int* a = new int[half];
    int* b = new int[_numProcs - 1 - half];
    int server[1];

    for (int i = 1; i <= half; i++)
        a[i] = i;
    for (int i = half + 1; i < _numProcs; i++)
        b[i - half - 1] = i;

    server[0] = 0;

    
    MPI_Comm_group(_world, &worldGroup);
    MPI_Group_incl(worldGroup, half, a, &AGroup); 
    MPI_Group_incl(worldGroup, _numProcs - 1 - half, b, &BGroup);

    MPI_Comm_create(_world, AGroup, &A);
    MPI_Comm_create(_world, BGroup, &B);

    if (_myId <= half && _myId > 0)
    {
        _partWorld = A;
    }
    else if (_myId > half)
    {
        _partWorld = B;
    }

    delete[] a;
    delete[] b;

}



