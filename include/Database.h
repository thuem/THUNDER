/*******************************************************************************
 * Author: Mingxu Hu, Bing Li
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <cstdio>
#include <string>
#include <cstring>
#include <sys/stat.h>

#include <sqlite3.h>

#include <mpi.h>

#include "Sqlite3Error.h"

#define MAX_LENGTH (1024 * 1024 * 128)

#define MASTER_RAMPATH         "/thuem/shm/dbMaster"
#define SLAVE_RAMPATH          "/thuem/shm/dbSlave"

using namespace std;

#define SQLITE3_CALLBACK [](void* data, int ncols, char** values, char** header)

#define MASTER_TMP_FILE(database, rank) \
    [&database, &rank]() mutable \
    { \
        sprintf(database, "%s/node%04d.db", MASTER_RAMPATH, rank); \
    }()

#define SLAVE_TMP_FILE(database) \
    [this, &database]() mutable \
    { \
        sprintf(database, "%s/node%04d.db", SLAVE_RAMPATH, _commRank); \
    }()

#define WRITE_FILE(filename, buf, len) \
    [&filename, &buf, &len]() \
    {\
        FILE* fd = fopen(filename, "w"); /* TODO Error Checking */ \
        fwrite(buf, len, 1, fd); /* TODO Error Checking */ \
        fclose(fd); \
    }()

#define READ_FILE(filename, buf) \
    [&filename, &buf]() \
    {\
        FILE* fd = fopen(filename, "r"); /* TODO Error Checking */ \
        fseek(fd, 0, SEEK_SET); \
        int size = fread(buf, 1, MAX_LENGTH, fd); /* TODO Error Checking */ \
        fclose(fd); \
        return size; \
    }()

enum Table
{
    Groups,
    Micrographs,
    Particles
};

class Database
{
    private:

        sqlite3_stmt* _stmtAppendGroup = NULL;
        sqlite3_stmt* _stmtAppendMicrograph = NULL;
        sqlite3_stmt* _stmtAppendParticle = NULL;

    protected:

        sqlite3* _db;

        int _commRank = 0;
        int _commSize = 1;

    public:

        Database();

        Database(const char database[]);

        ~Database();

        void setCommSize(int commSize);

        void setCommRank(int commRank);

        void openDatabase(const char database[]);

        void saveDatabase(sqlite3* database);

        void saveDatabase(const char database[]);

        void saveDatabase(const int rank);

        void createTableGroups();

        void createTableMicrographs();
        
        void createTableParticles();

        void appendGroup(const char name[],
                         const int id = -1);

        void appendMicrograph(const char name[],
                              const float voltage,
                              const float defocusU,
                              const float defocusV,
                              const float defocusAngle,
                              const float CA,
                              const int id = -1);

        void appendParticle(const char name[],
                            const int groupID,
                            const int micrographID);

        void update(const char database[],
                    const Table table);

        void prepareTmpFile();

        void receive();

        void send();

    protected:

        void split(int& startParticleID,
                   int& endParticleID,
                   int commRank) const;

    private:
        
        void setTempInMemory();
        /* store temp in memory */

        void finalizeStatement();

        void masterPrepareTmpFile();

        void masterReceive(const int rank);

        void slaveReceive();

        void masterSend(const int rank);

        void slaveSend();
};
