/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Siyuan Ren
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

#include <boost/noncopyable.hpp>

/**
 * The maximum buf size used in MPI environment. It is just a little bit smaller
 * than INT_MAX, and it can be divided by 2, 4, 8 and 16.
 * BTW, INT_MAX = 2147483647
 */
#define MPI_MAX_BUF 2000000000

/**
 * rank ID of master process
 */
#define MASTER_ID 0

/**
 * rank ID of the leader process of hemisphere A
 */
#define HEMI_A_LEAD 1

/**
 * rank IF of the leader process of hemisphere B
 */
#define HEMI_B_LEAD 2

/**
 * This macro is a short hand of a condition statement that the current process
 * is the master process.
 */
#define IF_MASTER if (_commRank == MASTER_ID)

/**
 * This macro is a short hand of a condition statement that the current process
 * is not the master process.
 */
#define NT_MASTER if (_commRank != MASTER_ID)

/**
 * This macro is a short hand of logging when the current process is the
 * master process.
 *
 * @param LEVEL  level of the log
 * @param LOGGER logger of the log
 */
#define MLOG(LEVEL, LOGGER) \
    IF_MASTER CLOG(LEVEL, LOGGER) << "MASTER: "

/**
 * This macro is a short hand of logging when the current process is the
 * leader process of hemisphere A.
 *
 * @param LEVEL  level of the log
 * @param LOGGER logger of the log
 */
#define ALOG(LEVEL, LOGGER) \
    if (_commRank == HEMI_A_LEAD) CLOG(LEVEL, LOGGER) << "A_LEAD: "

/**
 * This macro is a short hand of logging when the current process is the
 * leader process of hemisphere B.
 *
 * @param LEVEL  level of the log
 * @param LOGGER logger of the log
 */
#define BLOG(LEVEL, LOGGER) \
    if (_commRank == HEMI_B_LEAD) CLOG(LEVEL, LOGGER) << "B_LEAD: "

/**
 * This macro is a short hand of logging when the current process is not the 
 * master process. Along with this log, the rank ID of the current process 
 * will be shown.
 *
 * @param LEVEL  level of the log
 * @param LOGGER logger of the log
 */
#define ILOG(LEVEL, LOGGER) \
    NT_MASTER CLOG(LEVEL, LOGGER) << "RANK " << _commRank << ": "

class Parallel: private boost::noncopyable
{
    protected:

        /**
         * number of processes in MPI_COMM_WORLD
         */
        int _commSize;

        /**
         * the rank ID of the current process in MPI_COMM_WORLD
         */
        int _commRank;

        /**
         * communicator of hemisphere A(B)
         */
        MPI_Comm _hemi;

    public:

        /**
         * default constructor
         */
        Parallel();

        /**
         * default deconstructor
         */
        ~Parallel();

        /**
         * This function detects the number of processes in MPI_COMM_WORLD and
         * the rank ID of the current process in MPI_COMM_WORLD. Moreover, it
         * will assign all process in MPI_COMM_WORLD into three parts: master,
         * hemisphere A and hemisphere B.
         */
        void setMPIEnv();

        /**
         * This function inherits the MPI information by parameters.
         *
         * @param commSize the numbber of process in MPI_COMM_WORLD
         * @param commRank the rank ID of the current process in MPI_COMM_WORLD
         * @param hemi     the hemisphere of the current process
         */
        void setMPIEnv(const int commSize,
                       const int commRank,
                       const MPI_Comm& hemi);

        /**
         * This function returns whether the current process is the master
         * process or not.
         */
        bool isMaster() const;

        /**
         * This function returns whether the current process is in hemisphere A
         * or not.
         */
        bool isA() const;

        /**
         * This function returns whether the current process is in hemisphere B
         * or not.
         */
        bool isB() const;

        /**
         * This function returns the number of processes in MPI_COMM_WORLD.
         */
        int commSize() const;

        /**
         * This function sets the number of processes in MPI_COMM_WORLD.
         *
         * @param commSize the number of processes in MPI_COMM_WORLD
         */
        void setCommSize(const int commSize);

        /**
         * This function returns the rank ID of the current process in
         * MPI_COMM_WORLD.
         */
        int commRank() const;

        /**
         * This function sets the rank ID of the current process in
         * MPI_COMM_WORLD.
         *
         * @param commRank the rank ID of the current process in MPI_COMM_WORLD
         */
        void setCommRank(const int commRank);

        /**
         * This function returns the hemisphere of the current process.
         */
        MPI_Comm hemi() const;

        /**
         * This function sets the hemisphere of the current process.
         *
         * @param hemi the hemisphere of the current process
         */
        void setHemi(const MPI_Comm& hemi);
};

/**
 * This function displays MPI information of a Parallel object.
 *
 * @param parallel a Parallel object
 */
void display(const Parallel& parallel);

/**
 * This function is an overwrite of MPI_Recv function for large data
 * transporting.
 *
 * @param buf      the buffer area for receiving data
 * @param count    the number of the data
 * @param datatype the type of the data
 * @param source   the rank ID of the sending process in the communicator
 * @param tag      the tag of the data
 * @param comm     the communicator
 */
void MPI_Recv_Large(void* buf,
                    size_t count,
                    MPI_Datatype datatype,
                    int source,
                    int tag,
                    MPI_Comm comm);

/**
 * This function is an overwrite of MPI_Ssend function for large data
 * transporting.
 *
 * @param buf      the buffer area for sending data
 * @param count    the number of the data
 * @param datatype the type of the data
 * @param dest     the rank ID of the receiving process in the communicator
 * @param tag      the tag of the data
 * @param comm     the communicator
 */
void MPI_Ssend_Large(const void* buf,
                     size_t count,
                     MPI_Datatype datatype,
                     int dest,
                     int tag,
                     MPI_Comm comm);

/**
 * This function is an overwrite of MPI_Bcast function for large data
 * transporting.
 *
 * @param buf      the buffer area for broadcasting data
 * @param count    the number of the data
 * @param datatype the type of the data
 * @param dest     the rank ID of the root process in the communicator
 * @param comm     the communicator
 */
void MPI_Bcast_Large(void* buf,
                     size_t count,
                     MPI_Datatype datatype,
                     int root,
                     MPI_Comm comm);

/**
 * This function is an overwrite of MPI_Allreduce function for large data
 * transporting.
 *
 * @param buf      the buffer area of all-reducing data
 * @param count    the number of the data
 * @param datatype the type of the data
 * @param op       the operator of all-reducing
 * @param comm     the communicator
 */
void MPI_Allreduce_Large(void* buf,
                         size_t count,
                         MPI_Datatype datatype,
                         MPI_Op op,
                         MPI_Comm comm);

#endif // PARALLEL_H
