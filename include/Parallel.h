/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Siyuan Ren
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/
/** @file
 *  @author Hongkun Yu
 *  @author Mingxu Hu
 *  @author Siyuan Ren
 *  @author Huabin Ruan
 *  @version 1.4.11.180919
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Huabin Ruan | 2018/09/13 | 1.4.11.080913 | Add header for file and functions
 *  Mingxu Hu   | 2018/12/22 | 1.4.11.081222 | Add some emendation in the documentation
 *
 *  @brief Parallel.h encapsulates MPI related functions used for process partitioning, sending and receiving large size data, and so on.
  */

#ifndef PARALLEL_H
#define PARALLEL_H

#include <cstdio>
#include <mpi.h>
#include "Logging.h"
#include "Precision.h"
#include <boost/noncopyable.hpp>

/**
 * @brief the maximum buf size used in MPI environment. It is just a little bit smaller than INT_MAX(2147483647), and it can be divided by 2, 4, 8 and 16.
 */
#define MPI_MAX_BUF 2000000000

/**
 * @brief process ID of master process
 */
#define MASTER_ID 0

/**
 * @brief process ID of the leader process of hemisphere A
 */
#define HEMI_A_LEAD 1

/**
 * @brief process ID of the leader process of hemisphere B
 */
#define HEMI_B_LEAD 2

/**
 * @brief This macro is a short hand of a condition statement that the current process is the master process.
 */
#define IF_MASTER if (_commRank == MASTER_ID)

/**
 * @brief This macro is a short hand of a condition statement that the current process is not the master process.
 */
#define NT_MASTER if (_commRank != MASTER_ID)

/**
 * @brief This macro is a short hand of logging when the current process is the master process.
 *
 * [in] LEVEL  level of the log
 * [in] param LOGGER logger of the log
 */
#define MLOG(LEVEL, LOGGER) \
    IF_MASTER CLOG(LEVEL, LOGGER) << "MASTER: "

/**
 * @brief This macro is a short hand of logging when the current process is the leader process of hemisphere A.
 *
 * [in] LEVEL  level of the log
 * [in] LOGGER logger of the log
 */
#define ALOG(LEVEL, LOGGER) \
    if (_commRank == HEMI_A_LEAD) CLOG(LEVEL, LOGGER) << "A_LEAD: "

/**
 * @brief This macro is a short hand of logging when the current process is the leader process of hemisphere B.
 *
 * [in] LEVEL  level of the log
 * [in] LOGGER logger of the log
 */
#define BLOG(LEVEL, LOGGER) \
    if (_commRank == HEMI_B_LEAD) CLOG(LEVEL, LOGGER) << "B_LEAD: "

/**
 * @brief This macro is a short hand of logging when the current process is not the master process. Along with this log, the rank ID of the current process will be shown.
 *
 * [in] LEVEL  level of the log
 * [in] LOGGER logger of the log
 */
#define ILOG(LEVEL, LOGGER) \
    NT_MASTER CLOG(LEVEL, LOGGER) << "RANK " << _commRank << ": "


/**
 *  @brief Define a macro used for logging memory usage information in thunder.
 *  
 *  [in] msg string prefix of the message.
 */
#define CHECK_MEMORY_USAGE(msg) \
    do \
    { \
        long memUsageRM = memoryCheckRM(); \
        ALOG(INFO, "LOGGER_MEM") << msg << ", Physic Memory Usage : " << memUsageRM / MEGABYTE << "G"; \
        BLOG(INFO, "LOGGER_MEM") << msg << ", Physic Memory Usage : " << memUsageRM / MEGABYTE << "G"; \
    } while (0);

/**
 * @brief The Parallel class generates, stores and exchanges MPI information.
 *
 * This class generates, stores and exhanges MPI information as the number of process in MPI_COMM_WORLD, the rank ID of the current process in MPI_COMM_WORLD and the MPI communications such as hemispere A , hemisphere B and all slave processes.
 */
class Parallel: private boost::noncopyable
{

    protected:

    /**
     * @brief number of processes in MPI_COMM_WORLD
     */
    int _commSize;

    /**
     * @brief the rank ID of the current process in MPI_COMM_WORLD
     */
    int _commRank;

    /**
     * @brief communicator of hemisphere A(B)
     */
    MPI_Comm _hemi;

    /**
     * @brief communicator of all slave processes
     */
    MPI_Comm _slav;

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
     * @brief This function detects the number of processes in MPI_COMM_WORLD and the rank ID of the current process in MPI_COMM_WORLD. Moreover, it will assign all process in MPI_COMM_WORLD into three parts: master, hemisphere A and hemisphere B.
     */
    void setMPIEnv();

    /**
     * @brief This function inherits the MPI information by parameters.
     *
     * @param commSize the numbber of process in MPI_COMM_WORLD
     * @param commRank the rank ID of the current process in MPI_COMM_WORLD
     * @param hemi     the hemisphere of the current process
     */
    void setMPIEnv(const int commSize,  /**< [in] the number of process in MPI_COMM_WORLD. */
                  const int commRank,   /**< [in] the process ID of the current process in MPI_COMM_WORLD. */
                  const MPI_Comm &hemi, /**< [in] the hemisphere of the current process. */
                  const MPI_Comm &slav  /**< [in] the slave communitor. */
                  );

    /**
     * @brief Check whether the current process is the master process or not.
     */
    bool isMaster() const;

    /**
     * @brief Check whether the current process is in hemisphere A or not.
     */
    bool isA() const;

    /**
     * @brief Check whether the current process is in hemisphere B or not.
     */
    bool isB() const;

    /**
     * @brief Get the number of processes in MPI_COMM_WORLD.
     */
    int commSize() const;

    /**
     * @brief Sets the number of processes in MPI_COMM_WORLD.
     *
     */
    void setCommSize(const int commSize /**< [in] the number of processes in MPI_COMM_WORLD. */);

    /**
     * @brief Get the process ID of the current process in MPI_COMM_WORLD.
     */
    int commRank() const;

    /**
     * @brief Sets the process ID of the current process in MPI_COMM_WORLD.
     *
     */
    void setCommRank(const int commRank /**< [in] process ID to be set. */);

    /**
     * @brief Get the hemisphere of the current process.
     */
    MPI_Comm hemi() const;

    /**
     * @brief Sets the hemisphere of the current process.
     */
    void setHemi(const MPI_Comm &hemi /**< [in] the hemisphere that the current process belongs to. */);



    /**
     *  @brief Get the communitor that the slave process belongs to.
     */
    MPI_Comm slav() const;


    /**
     *  @brief Set the communicator for the slave process.
     */
    void setSlav(const MPI_Comm &slav /**< [in] the communicator for the slave process.*/);
};


/**
 * @brief This function can be used for receiving large size(>2GB) data. 
 */
void MPI_Recv_Large(void *buf,             /**< [out] the data buffer used to receive data. */
                    size_t count,          /**< [in]  the number of data elements to receive. */
                    MPI_Datatype datatype, /**< [in]  the type of the received data elements. */
                    int source,            /**< [in]  the process ID of the sending process in the communicator. */
                    int tag,               /**< [in]  the unique tag used for identifying the data. */
                    MPI_Comm comm          /**< [in]  the communicator that the sending/receiving processes belongs to. */
                   );

/**
 * @brief This function can be used for sending large size(>2GB) data. 
 */
void MPI_Ssend_Large(const void *buf,       /**< [in] the data elements to be sent. */
                     size_t count,          /**< [in] the number of data elements to be sent. */
                     MPI_Datatype datatype, /**< [in] the type of the  data elements to be sent. */
                     int dest,              /**< [in] the process ID of the received process in the communicator. */
                     int tag,               /**< [in] the unique tag used for identifying the data. */
                     MPI_Comm comm          /**< [in] the communicator that the sending/receiving processes belongs to. */
                    );

/**
 * @brief  This function can be used for broadcasting large size(>2GB) data
 */
void MPI_Bcast_Large(void *buf,             /**< [in] the data elements used for broadcasting. */
                     size_t count,          /**< [in] the number of data elements to be broadcasted. */
                     MPI_Datatype datatype, /**< [in] the type of the  data elements to be broadcasted. */
                     int root,              /**< [in] the process ID of the broadcasting process in the communicator. */
                     MPI_Comm comm          /**< [in] the communicator that the sending/receiving processes belongs to. */
                     );
/**
 * @brief This function is used for all reducing operation for large size(>2GB) data
 *
 * @param buf      the buffer area of all-reducing data
 * @param count    the number of the data
 * @param datatype the type of the data
 * @param op       the operator of all-reducing
 * @param comm     the communicator
 */
void MPI_Allreduce_Large(void *buf,             /**< [in] the data buffer used for saving all reducing result. */
                         size_t count,          /**< [in] the number of data elements to be reduced. */
                         MPI_Datatype datatype, /**< [in] the type of the  data elements to be reduced. */
                         MPI_Op op,             /**< [in] the operation to be performed in all reducing */ 
                         MPI_Comm comm        /**< [in] the communicator that the all reducing processes belongs to. */
                        );

#endif // PARALLEL_H
