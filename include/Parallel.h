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
//#include <glog/logging.h>

#define MASTER_ID 0
#define HEMI_A_LEAD 1
#define HEMI_B_LEAD 2

#define IF_MASTER if (_commRank == MASTER_ID)
#define NT_MASTER if (_commRank != MASTER_ID)

#define MLOG(LEVEL) IF_MASTER LOG(LEVEL) << "MASTER: "
#define ALOG(LEVEL) if (_commRank == HEMI_A_LEAD) LOG(LEVEL) << "A_LEAD: "
#define BLOG(LEVEL) if (_commRank == HEMI_B_LEAD) LOG(LEVEL) << "B_LEAD: "
#define ILOG(LEVEL) LOG(LEVEL) << "RANK " << _commRank << ": ";

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
