
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
    MPI_Group AWorkGroup, BWorkGroup;

    MPI_Comm A, B;
    MPI_Comm Aworker, Bworker;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(_world, &_numProcs);
    MPI_Comm_rank(_world, &_myId);

    int half = _numProcs / 2;
    int* a = new int[half];
    int* b = new int[_numProcs - half];
    for (int i = 0; i < half; i++)
        a[i] = i;
    for (int i = half; i < _numProcs; i++)
        b[i - half] = half;

    MPI_Comm_group(_world, &worldGroup);
    MPI_Group_excl(worldGroup, half, a, &BGroup);
    MPI_Group_excl(worldGroup, _numProcs - half, b, &AGroup); 
    MPI_Comm_create(_world, AGroup, &A);
    MPI_Comm_create(_world, BGroup, &B);

    int aserver[1], bserver[1];
    aserver[0] = 0;
    bserver[0] = half;

    MPI_Group_excl(AGroup, 1, aserver, &AWorkGroup);
    MPI_Group_excl(BGroup, 1, bserver, &BWorkGroup); 
    MPI_Comm_create(A, AWorkGroup, &Aworker);
    MPI_Comm_create(B, BWorkGroup, &Bworker);

    if (_myId < half)
    {
        _partWorld = A;
        _serverId = 0;
        if (_myId != _serverId)
        {
            _partWorldW = Aworker;
            MPI_Comm_rank(Aworker, &_workerId);
            MPI_Comm_size(Aworker, &_numWorkers);
        }
    }
    else
    {
        _partWorld = B;
        _serverId = half;
        if (_myId != _serverId)
        {
            _partWorldW = Bworker;
            MPI_Comm_rank(Bworker, &_workerId);
            MPI_Comm_size(Bworker, &_numWorkers);
        }
    }

    delete[] a;
    delete[] b;

}



