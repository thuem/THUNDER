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

#include "Logging.h"

/**
 * The maximum buf size used in MPI environment. It is just a little bit smaller
 * than INT_MAX, and it can be divided by 2, 4, 8 and 16.
 * BTW, INT_MAX = 2147483647
 */
#define MPI_MAX_BUF 2000000000

#define MASTER_ID 0

#define HEMI_A_LEAD 1

#define HEMI_B_LEAD 2

#define IF_MASTER if (_commRank == MASTER_ID)

#define NT_MASTER if (_commRank != MASTER_ID)

#define MLOG(LEVEL, LOGGER) \
    IF_MASTER CLOG(LEVEL, LOGGER) << "MASTER: "

#define ALOG(LEVEL, LOGGER) \
    if (_commRank == HEMI_A_LEAD) CLOG(LEVEL, LOGGER) << "A_LEAD: "

#define BLOG(LEVEL, LOGGER) \
    if (_commRank == HEMI_B_LEAD) CLOG(LEVEL, LOGGER) << "B_LEAD: "

#define ILOG(LEVEL, LOGGER) \
    CLOG(LEVEL, LOGGER) << "RANK " << _commRank << ": "

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

        bool isA() const;

        bool isB() const;

        int commSize() const;

        void setCommSize(const int commSize);

        int commRank() const;

        void setCommRank(const int commRank);

        MPI_Comm hemi() const;

        void setHemi(const MPI_Comm& hemi);
};

void display(const Parallel& parallel);

void MPI_Recv_Large(void* buf,
                    size_t count,
                    MPI_Datatype datatype,
                    int source,
                    int tag,
                    MPI_Comm comm);

void MPI_Ssend_Large(const void* buf,
                     size_t count,
                     MPI_Datatype datatype,
                     int dest,
                     int tag,
                     MPI_Comm comm);

void MPI_Bcast_Large(void* buf,
                     size_t count,
                     MPI_Datatype datatype,
                     int root,
                     MPI_Comm comm);

void MPI_Allreduce_Large(const void* sendbuf,
                         void* recvbuf,
                         size_t count,
                         MPI_Datatype datatype,
                         MPI_Op op,
                         MPI_Comm comm);

#endif // PARALLEL_H
