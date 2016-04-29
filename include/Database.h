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


#include "Random.h"
#include "Parallel.h"
#include "SQLWrapper.h"

#define PARTICLE_MODE 0
#define MICROGRAPH_MODE 1

#define DB_ID_LENGTH 20

#define MAX_LENGTH (1024 * 1024 * 128)

#define SQLITE3_CALLBACK [](void* data, int ncols, char** values, char** header)

#define MASTER_TMP_FILE(database, rank) \
    [this, &database](const int _rank) mutable \
    { \
        sprintf(database, "/tmp/%s/m/%04d.db", _ID, _rank); \
    }(rank)

#define SLAVE_TMP_FILE(database) \
    [this, &database]() mutable \
    { \
        sprintf(database, "/tmp/%s/s/%04d.db", _ID, _commRank); \
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

using namespace std;

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

        int _mode = PARTICLE_MODE;

        sql::Statement _stmtAppendGroup;
        sql::Statement _stmtAppendMicrograph;
        sql::Statement _stmtAppendParticle;

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
        
        int nParticle() const;
        /* number of particles */

        int nMicrograph() const;
        /* number of micrographs */

        int nGroup() const;

        void update(const char database[],
                    const Table table);

        void prepareTmpFile();

        void gather();

        void scatter();

    protected:

        void split(int& start,
                   int& end,
                   int commRank) const;

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
