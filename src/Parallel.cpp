/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Particle.h"
#include "Parallel.h"

Parallel::Parallel() {}

Parallel::~Parallel() {}

/***
Parallel::Parallel(int argc,
                   char* argv[])
{
    init(argc, argv);
}
***/

void Parallel::setMPIEnv(int argc,
                         char* argv[])
{
    MPI_Group wGroup, aGroup, bGroup;

    MPI_Comm A, B;
    
    // MPI_Init(&argc, &argv) -> This should be in main.
    MPI_Comm_size(MPI_COMM_WORLD, &_commSize);
    MPI_Comm_size(MPI_COMM_WORLD, &_commRank);
    /***
    MPI_Comm_size(_world, &_numProcs);
    MPI_Comm_rank(_world, &_myId);
    ***/

    int sizeA = (_commSize - 1) / 2;
    int sizeB = _commSize - 1 - sizeA;

    int* a = new int[sizeA];
    int* b = new int[sizeB];

    /***
    int* a = new int[half];
    int* b = new int[_numProcs - 1 - half];
    int server[1];
    ***/
    for (int i = 0; i < sizeA; i++) a[i] = i + 1;
    for (int i = 0; i < sizeB; i++) b[i] = i + 1 + sizeA;

    /***
    for (int i = 1; i <= half; i++)
        a[i] = i;
    for (int i = half + 1; i < _numProcs; i++)
        b[i - half - 1] = i;
        ***/

    /***
    server[0] = 0;
    ***/
    
    MPI_Comm_group(MPI_COMM_WORLD, &wGroup);
    MPI_Group_incl(wGroup, sizeA, a, &aGroup);
    MPI_Group_incl(wGroup, sizeB, b, &bGroup);
    /***
    MPI_Group_incl(worldGroup, half, a, &AGroup); 
    MPI_Group_incl(worldGroup, _numProcs - 1 - half, b, &BGroup);
    ***/
    MPI_Comm_create(MPI_COMM_WORLD, aGroup, &A);
    MPI_Comm_create(MPI_COMM_WORLD, aGroup, &B);

    if ((_commRank >= 1) && (_commRank < 1 + sizeA)) _hemi = A;
    else if (_commRank >= 1 + sizeA) _hemi = B;
    /***
    if (_myId <= half && _myId > 0)
    {
        _partWorld = A;
    }
    else if (_myId > half)
    {
        _partWorld = B;
    }
    ***/

    delete[] a;
    delete[] b;
}

void Parallel::setMPIEnv(const int commSize,
                         const int commRank,
                         const MPI_Comm& hemi)
{
    setCommSize(commSize);
    setCommRank(commRank);
    setHemi(hemi);
}

bool Parallel::isMaster() const
{
    return (_commRank == SERVER_ID);
}

int Parallel::commSize() const
{
    return _commSize;
}

void Parallel::setCommSize(const int commSize)
{
    _commSize = commSize;
}

int Parallel::commRank() const
{
    return _commRank;
}

void Parallel::setCommRank(const int commRank)
{
    _commRank = commRank;
}

void Parallel::setHemi(const MPI_Comm& hemi)
{
    _hemi = hemi;
}
