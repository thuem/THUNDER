/*******************************************************************************
 * Author: Mingxu Hu, Bing Li
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef DATABASE_H
#define DATABASE_H

#include <cstdio>
#include <string>
#include <cstring>
#include <sys/stat.h>

#include <cerrno>
#include <cstdio>

#include "Random.h"
#include "Parallel.h"
#include "SQLWrapper.h"

#define PARTICLE_MODE 0
#define MICROGRAPH_MODE 1

#define DB_ID_LENGTH 20

#define MAX_LENGTH (1024 * 1024 * 128)

#define SQLITE3_CALLBACK [](void* data, int ncols, char** values, char** header)

#define MASTER_TMP_FILE(database, rank) snprintf((database), sizeof(database), "/tmp/%s/m/%04d.db", _ID, (rank))

#define SLAVE_TMP_FILE(database) snprintf(database, sizeof(database), "/tmp/%s/s/%04d.db", _ID, _commRank)

inline void WRITE_FILE(const char* filename, void* buf, size_t len)
{
    FILE* fd = fopen(filename, "w");
    if (fd == NULL)
        CLOG(FATAL, "LOGGER_SYS") << "Can Not Open Sqlite3 File : "  << filename << strerror(errno);
    if (fwrite(buf, len, 1, fd) != 1)
        CLOG(FATAL, "LOGGER_SYS") << "Write " << filename << " failed: " << strerror(errno);
    fclose(fd);
}

inline int READ_FILE(const char* filename, void* buf)
{
    FILE* fd = fopen(filename, "r");
    if (fd == NULL)
        CLOG(FATAL, "LOGGER_SYS") << "Can Not Open Sqlite3 File : "  << filename << strerror(errno);
    int size = fread(buf, 1, MAX_LENGTH, fd);
    if (size <= 0)
        CLOG(FATAL, "LOGGER_SYS") << "Read " << filename << " failed: " << strerror(errno);
    if (size >= MAX_LENGTH)
        CLOG(FATAL, "LOGGER_SYS") << "Sqlite3 file " << filename << " exceeds maximum length " << MAX_LENGTH;
    fclose(fd);
    return size;
 }



enum Table
{
    Groups,
    Micrographs,
    Particles
};

class Database : public Parallel
{
    private:

        char _ID[DB_ID_LENGTH + 1];

        int _mode;

        sql::Statement _stmtAppendGroup;
        sql::Statement _stmtAppendMicrograph;
        sql::Statement _stmtAppendParticleS;
        sql::Statement _stmtAppendParticleL;

    protected:

        sql::DB _db;

    public:

        Database();

        Database(const char database[]);

        ~Database();

        sql::DB expose() { return _db; }

        void bcastID();
        /* generate and broadcast an unique ID */

        int mode() const;

        void setMode(const int mode);

        void openDatabase(const char database[]);

        void saveDatabase(sql::DB database);

        void saveDatabase(const char database[]);

        void saveDatabase(const int rank);

        void createTables();

        void createTableGroups();

        void createTableMicrographs();
        
        void createTableParticles();

        void appendGroup(const char name[],
                         const int id = -1);

        void appendMicrograph(const char name[],
                              const double voltage,
                              const double defocusU,
                              const double defocusV,
                              const double defocusAngle,
                              const double CS,
                              const int id = -1);

        void appendParticle(const char name[],
                            const int groupID,
                            const int micrographID);

        void appendParticle(const char name[],
                            const int groupId,
                            const int micrographId,
                            const double voltage,
                            const double defocusU,
                            const double defocusV,
                            const double defocusAngle,
                            const double Cs);
        
        int nParticle();
        /* number of particles */

        int nMicrograph();
        /* number of micrographs */

        int nGroup();

        void update(const char database[],
                    const Table table);

        void prepareTmpFile();

        void gather();

        void scatter();

    protected:

        void split(int& start,
                   int& end,
                   int commRank);

    private:
        
        void setTempInMemory();
        /* store temp in memory */

        void finalizeStatement();

        void masterPrepareTmpFile();

        void slavePrepareTmpFile();

        void masterReceive(const int rank);

        void slaveReceive();

        void masterSend(const int rank);

        void slaveSend();
};

#endif // DATABASE_H
