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

#include "Sqlite3Error.h"
#include "Random.h"
#include "Parallel.h"

#define PARTICLE_MOD 0
#define MICROGRAPH_MOD 1

#define DB_ID_LENGTH 20

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

class Database : public Parallel
{
    private:

        char _ID[DB_ID_LENGTH + 1];

        int _mode = PARTICLE_MOD;

        sqlite3_stmt* _stmtAppendGroup = NULL;
        sqlite3_stmt* _stmtAppendMicrograph = NULL;
        sqlite3_stmt* _stmtAppendParticle = NULL;

    protected:

        sqlite3* _db;

    public:

        Database();

        Database(const char database[]);

        ~Database();

        void BcastID();
        /* generate and broadcast an unique ID */

        int mode() const;

        void setMode(const int mode);

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
                              const double voltage,
                              const double defocusU,
                              const double defocusV,
                              const double defocusAngle,
                              const double CA,
                              const int id = -1);

        void appendParticle(const int groupID,
                            const int micrographID);
        
        int nParticle() const;
        /* number of particles */

        int nMicrograph() const;
        /* number of micrographs */

        void update(const char database[],
                    const Table table);

        void prepareTmpFile();

        void receive();

        void send();

    protected:

        void split(int& start,
                   int& end,
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
