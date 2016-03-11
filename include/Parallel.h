/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef PARALLEL_H
#define PARALLEL_H

#include <cstdio>
#include <mpi.h>

#define MASTER_ID 0

class Parallel
{
    protected:

        int _commSize;
        int _commRank;

        MPI_Comm _hemi;

    public:

        Parallel();

        ~Parallel();

        void setMPIEnv();

        void setMPIEnv(const int commSize,
                       const int commRank,
                       const MPI_Comm& hemi);

        bool isMaster() const;

        int commSize() const;

        void setCommSize(const int commSize);

        int commRank() const;

        void setCommRank(const int commRank);

        MPI_Comm hemi() const;

        void setHemi(const MPI_Comm& hemi);
};

void display(const Parallel& parallel);

#endif // PARALLEL_H
