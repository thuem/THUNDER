/*******************************************************************************
 * Author: Mingxu Hu, Bing Li
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Database.h"

Database::Database()
{
    SQLITE3_HANDLE_ERROR(sqlite3_open(":memory:", &_db));

    setTempInMemory();
}

Database::Database(const char database[])
{
    openDatabase(database);
}

Database::~Database()
{
    // finalize all statements
    finalizeStatement();

    // disconnect the database
    SQLITE3_HANDLE_ERROR(sqlite3_close(_db));
}

int Database::mode() const
{
    return _mode;
}

void Database::setMode(const int mode)
{
    _mode = mode;
}

void Database::openDatabase(const char database[])
{
    // connect to the data base
    SQLITE3_HANDLE_ERROR(sqlite3_open(database, &_db));
    
    setTempInMemory();
}

void Database::saveDatabase(sqlite3* database)
{
    sqlite3_backup* backupDB = sqlite3_backup_init(database,
                                                   "main",
                                                   _db,
                                                   "main");

    if (backupDB)
    {
        SQLITE3_HANDLE_ERROR(sqlite3_backup_step(backupDB, -1));
        SQLITE3_HANDLE_ERROR(sqlite3_backup_finish(backupDB));
    }
}

void Database::saveDatabase(const char database[])
{
    // open dst database
    sqlite3* dstDB;
    SQLITE3_HANDLE_ERROR(sqlite3_open(database, &dstDB));

    // save to dst database
    saveDatabase(dstDB);

    // close dst database
    SQLITE3_HANDLE_ERROR(sqlite3_close(dstDB));
}

void Database::saveDatabase(const int rank)
{
    char database[64];

    string sql;

    sprintf(database, "%s/node%04d.db", MASTER_RAMPATH, rank);  

    sql = "attach database '" +  string(database) + "' as dst;";
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, sql.c_str(), NULL, NULL, NULL));

    // get the ID of particles of process commRank
    int start, end;
    split(start, end, rank);

    if (_mode == PARTICLE_MOD)
        sql = "insert into dst.groups select distinct groups.* from \
               groups, particles where \
               (particles.groupID = groups.ID) and \
               (particles.ID >= ?1) and (particles.ID <= ?2); \
               insert into dst.micrographs \
               select distinct micrographs.* from \
               micrographs, particles where \
               particles.micrographID = micrographs.ID and \
               (particles.ID >= ?1) and (particles.ID <= ?2); \
               insert into dst.particles select * from particles \
               where (ID >= ?1) and (ID <= ?2);";
    else if (_mode == MICROGRAPH_MOD)
    {
        // TODO: change to proper SQL
        sql = "insert into dst.micrographs select * from micrographs \
               where (ID >= ?1) and (ID <= ?2); \
               insert into dst.particles select particles.* from \
               micrographs, particles where \
               particles.micrographID = micrographs.ID and \
               (micrographs.ID >= ?1) and (micrographs.ID <= ?2);";
        /***
        sql = "insert into dst.groups select distinct groups.* from \
               groups, particles where \
               (particles.groupID = groups.ID) and \
               (particles.ID >= ?1) and (particles.ID <= ?2); \
               insert into dst.micrographs \
               select distinct micrographs.* from \
               micrographs, particles where \
               particles.micrographID = micrographs.ID and \
               (particles.ID >= ?1) and (particles.ID <= ?2); \
               insert into dst.particles select * from particles \
               where (ID >= ?1) and (ID <= ?2);";
               ***/
    }

    sqlite3_stmt* _stmtSaveDatabase;
    const char* tail;
    while (sqlite3_complete(sql.c_str()))
    {
      
        SQLITE3_HANDLE_ERROR(sqlite3_prepare(_db,
                                             sql.c_str(),
                                             -1,
                                             &_stmtSaveDatabase,
                                             &tail));
        SQLITE3_HANDLE_ERROR(sqlite3_bind_int(_stmtSaveDatabase, 1, start));
        SQLITE3_HANDLE_ERROR(sqlite3_bind_int(_stmtSaveDatabase, 2, end));
        SQLITE3_HANDLE_ERROR(sqlite3_step(_stmtSaveDatabase));
        SQLITE3_HANDLE_ERROR(sqlite3_finalize(_stmtSaveDatabase));
        sql = string(tail);
    }

    sql = "detach database dst;";
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, sql.c_str(), NULL, NULL, NULL));
}

void Database::createTableGroups()
{
    SQLITE3_HANDLE_ERROR(
            sqlite3_exec(_db,
                         "create table groups(ID integer primary key, \
                                              Name text);",
                         NULL, NULL, NULL)); 

    const char sql[] = "insert into groups values (?, ?);";
    SQLITE3_HANDLE_ERROR(
            sqlite3_prepare_v2(_db,
                               sql,
                               strlen(sql),
                               &_stmtAppendGroup,
                               NULL));
}

void Database::createTableMicrographs()
{
    SQLITE3_HANDLE_ERROR(
            sqlite3_exec(_db,
                         "create table micrographs(ID integer primary key, \
                                                   Name text, \
                                                   Voltage real not null, \
                                                   DefocusU real not null, \
                                                   DefocusV real not null, \
                                                   DefocusAngle real not null, \
                                                   CA real not null);",
                         NULL, NULL, NULL));

    const char sql[] = "insert into micrographs \
                        values (?, ?, ?, ?, ?, ?, ?)";
    SQLITE3_HANDLE_ERROR(
            sqlite3_prepare_v2(_db,
                               sql,
                               strlen(sql),
                               &_stmtAppendMicrograph,
                               NULL));
}

void Database::createTableParticles()
{
    SQLITE3_HANDLE_ERROR(
            sqlite3_exec(_db,
                         "create table particles( \
                                  ID integer primary key, \
                                  GroupID integer not null, \
                                  MicrographID integer not null);",
                         NULL, NULL, NULL));
    
    const char sql[] = "insert into particles (GroupID, MicrographID) \
                        values (?, ?)";
    SQLITE3_HANDLE_ERROR(
            sqlite3_prepare_v2(_db,
                               sql,
                               strlen(sql),
                               &_stmtAppendParticle,
                               NULL));
}

void Database::appendGroup(const char name[],
                           const int id)
{
    if (id != -1)
        SQLITE3_HANDLE_ERROR(sqlite3_bind_int(_stmtAppendGroup, 1, id));
    else
        SQLITE3_HANDLE_ERROR(sqlite3_bind_null(_stmtAppendGroup, 1));

    SQLITE3_HANDLE_ERROR(
            sqlite3_bind_text(_stmtAppendGroup,
                              2,
                              name,
                              strlen(name),
                              SQLITE_TRANSIENT));
    SQLITE3_HANDLE_ERROR(sqlite3_step(_stmtAppendGroup));
    SQLITE3_HANDLE_ERROR(sqlite3_reset(_stmtAppendGroup));
}

void Database::appendMicrograph(const char name[],
                                const double voltage,
                                const double defocusU,
                                const double defocusV,
                                const double defocusAngle,
                                const double CA,
                                const int id)
{
    if (id != -1)
        SQLITE3_HANDLE_ERROR(sqlite3_bind_int(_stmtAppendMicrograph, 1, id));
    else
        SQLITE3_HANDLE_ERROR(sqlite3_bind_null(_stmtAppendMicrograph, 1));

    SQLITE3_HANDLE_ERROR(
            sqlite3_bind_text(_stmtAppendMicrograph,
                              2,
                              name,
                              strlen(name),
                              SQLITE_TRANSIENT));
    SQLITE3_HANDLE_ERROR(
            sqlite3_bind_double(_stmtAppendMicrograph,
                                3,
                                voltage));
    SQLITE3_HANDLE_ERROR(
            sqlite3_bind_double(_stmtAppendMicrograph,
                                4,
                                defocusU));
    SQLITE3_HANDLE_ERROR(
            sqlite3_bind_double(_stmtAppendMicrograph,
                                5,
                                defocusV));
    SQLITE3_HANDLE_ERROR(
            sqlite3_bind_double(_stmtAppendMicrograph,
                                6,
                                defocusAngle));
    SQLITE3_HANDLE_ERROR(
            sqlite3_bind_double(_stmtAppendMicrograph,
                                7,
                                CA));
    SQLITE3_HANDLE_ERROR(sqlite3_step(_stmtAppendMicrograph));
    SQLITE3_HANDLE_ERROR(sqlite3_reset(_stmtAppendMicrograph));
}

void Database::appendParticle(const int groupID,
                              const int micrographID)
{
    SQLITE3_HANDLE_ERROR(
            sqlite3_bind_int(_stmtAppendParticle,
                             1,
                             groupID));
    SQLITE3_HANDLE_ERROR(
            sqlite3_bind_int(_stmtAppendParticle,
                             2,
                             micrographID));
    SQLITE3_HANDLE_ERROR(sqlite3_step(_stmtAppendParticle));
    SQLITE3_HANDLE_ERROR(sqlite3_reset(_stmtAppendParticle));
}

int Database::nParticle() const
{
    int size;

    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db,
                                      "select count(*) from particles",
                                      SQLITE3_CALLBACK
                                      {
                                          *((int*)data) = atoi(values[0]);
                                          return 0;    
                                      },
                                      &size,
                                      NULL));
}

int Database::nMicrograph() const
{
    int size;

    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db,
                                      "select count(*) from micrographs",
                                      SQLITE3_CALLBACK
                                      {
                                          *((int*)data) = atoi(values[0]);
                                          return 0;    
                                      },
                                      &size,
                                      NULL));
}

void Database::update(const char database[],
                      const Table table)
{
    string sql = "attach database '" + string(database) + "' as src";
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, sql.c_str(), NULL, NULL, NULL));

    switch (table)
    {
        case Groups:
            sql = "replace into particles select * from src.particles";
            SQLITE3_HANDLE_ERROR(sqlite3_exec(_db,
                                              sql.c_str(),
                                              NULL, NULL, NULL));
            break;

        case Micrographs:
            sql = "replace into micrographs select * from src.micrographs";
            SQLITE3_HANDLE_ERROR(sqlite3_exec(_db,
                                              sql.c_str(),
                                              NULL, NULL, NULL));
            break;

        case Particles:
            sql = "replace into particles select * from src.particles";
            SQLITE3_HANDLE_ERROR(sqlite3_exec(_db,
                                              sql.c_str(),
                                              NULL, NULL, NULL));
            break;
    }

    sql = "detach database src";
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, sql.c_str(), NULL, NULL, NULL));
}

void Database::prepareTmpFile()
{
    if (_commRank == 0)
        masterPrepareTmpFile();
}

void Database::receive()
{
    if (_commRank == 0)
        for (int i = 1; i < _commSize; i++)
            masterReceive(i);
    else
        slaveSend();
}

void Database::send()
{
    if (_commRank == 0)
        for (int i = 1; i < _commSize; i++)
        {
            saveDatabase(i);
            masterSend(i);
        }
    else
        slaveReceive();
}

void Database::split(int& start,
                     int& end,
                     const int commRank) const
{
    int size;
    if (_mode == PARTICLE_MOD)
        size = nParticle();
    else if (_mode == MICROGRAPH_MOD)
        size = nMicrograph();

    int piece = size / (_commSize - 1);

    if (commRank <= size % (_commSize - 1))
    {
        start = (piece + 1) * (commRank - 1) + 1;
        end = start + (piece + 1);
    }
    else
    {
        start = piece * (commRank - 1) + size % (_commSize - 1) + 1;
        end = start + piece;
    }
}

void Database::setTempInMemory()
{
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db,
                                      "pragma temp_store = memory",
                                      NULL, NULL, NULL));
}

void Database::finalizeStatement()
{
    if (_stmtAppendGroup)
        SQLITE3_HANDLE_ERROR(sqlite3_finalize(_stmtAppendGroup));

    if (_stmtAppendMicrograph)
        SQLITE3_HANDLE_ERROR(sqlite3_finalize(_stmtAppendMicrograph));

    if (_stmtAppendParticle)
        SQLITE3_HANDLE_ERROR(sqlite3_finalize(_stmtAppendParticle));
}

void Database::masterPrepareTmpFile()
{
    if (_commRank != 0) return;

    // open dst database
    sqlite3* dstDB;

    char emptyDb[256];
    char nodeDb[256];
    char syscmd[128];
    
    string sql;

    // remove struct.db at tempory file direction
    sprintf(emptyDb, "%s/struct.db", MASTER_RAMPATH);
    sprintf(syscmd, "rm -f %s", emptyDb);  
    system(syscmd); 
    sprintf(syscmd, "touch %s", emptyDb);
    system(syscmd); 

    // create a database struct for each node
    sql = "attach database '" + string(emptyDb) + "' as dst";
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, sql.c_str(), NULL, NULL, NULL));

    sql= "create table dst.groups as select * from groups where 1=0";
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, sql.c_str(), NULL, NULL, NULL));

    sql= "create table dst.micrographs as select * from micrographs where 1=0";
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, sql.c_str(), NULL, NULL, NULL));

    sql= "create table dst.particles as select * from particles where 1=0";
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, sql.c_str(), NULL, NULL, NULL));

    sql = "detach database dst";
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, sql.c_str(), NULL, NULL, NULL));

    for (int i = 1; i < _commSize; i++)
    {
        sprintf(nodeDb, "%s/node%04d.db", MASTER_RAMPATH, i); 
        system(syscmd);         
        sprintf(syscmd, "cp %s %s", emptyDb, nodeDb);  
        system(syscmd);           
    }
}

void Database::masterReceive(const int rank)
{
    if (_commRank != 0) return;
    
    MPI_Status status;

    char database[128];
    MASTER_TMP_FILE(database, rank);

    char* buf = new char[MAX_LENGTH];

    MPI_Recv(buf, MAX_LENGTH, MPI_BYTE, rank, 0, MPI_COMM_WORLD, &status); 

    int len;
    MPI_Get_count(&status, MPI_BYTE, &len);

    WRITE_FILE(database, buf, len);

    delete[] buf;

    update(database, Particles);    
}

void Database::slaveReceive()
{
    if (_commRank == 0) return;
    
    MPI_Status status;

    char database[128];
    SLAVE_TMP_FILE(database);

    char* buf = new char[MAX_LENGTH];

    MPI_Recv(buf, MAX_LENGTH, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status); 

    int len;
    MPI_Get_count(&status, MPI_BYTE, &len);

    WRITE_FILE(database, buf, len);

    delete[] buf;

    SQLITE3_HANDLE_ERROR(sqlite3_close(_db));
    openDatabase(database);    
}

void Database::masterSend(const int rank)
{
    if (_commRank != 0) return;

    char database[128];
    MASTER_TMP_FILE(database, rank);

    char* buf = new char[MAX_LENGTH];

    int len = READ_FILE(database, buf);
    MPI_Ssend(buf, len, MPI_BYTE, rank, 0, MPI_COMM_WORLD);
   
    delete[] buf;
}

void Database::slaveSend()
{
    if (_commRank == 0) return;

    char database[128];
    SLAVE_TMP_FILE(database);

    char* buf = new char[MAX_LENGTH];

    int len = READ_FILE(database, buf);
    MPI_Ssend(buf, len, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    
    delete[] buf;
}
