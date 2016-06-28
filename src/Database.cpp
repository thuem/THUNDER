/*******************************************************************************
 * Author: Mingxu Hu, Bing Li, Siyuan Ren
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Database.h"

Database::Database()
{
    _db = sql::DB(":memory:", 0);

    setTempInMemory();
}

Database::Database(const char database[])
{
    openDatabase(database);
}

Database::~Database() {}

void Database::bcastID()
{
    auto engine = get_random_engine();
    MLOG(INFO, "LOGGER_INIT") << "Generating an Unique ID of Database";
    IF_MASTER
    {
        for (int i = 0; i < DB_ID_LENGTH; i++)
            _ID[i] = (char)(gsl_rng_uniform_int(engine, 26) + 65);
        _ID[DB_ID_LENGTH] = '\0';
    }
    MLOG(INFO, "LOGGER_INIT") << "ID is " << _ID;

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_INIT") << "Broadcasting the Unique ID of Database";
    MPI_Bcast(_ID, DB_ID_LENGTH + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
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
    _db = sql::DB(database, 0);

    setTempInMemory();
}

void Database::saveDatabase(sql::DB database)
{
    sqlite3_backup* backupDB = sqlite3_backup_init(database.getNativeHandle(),
                                                   "main",
                                                   _db.getNativeHandle(),
                                                   "main");

    if (backupDB)
    {
        int rc = sqlite3_backup_step(backupDB, -1);
        if (rc != SQLITE_OK && rc != SQLITE_DONE)
        {
            sqlite3_backup_finish(backupDB);
            throw sql::Exception(rc);
        }
        rc = sqlite3_backup_finish(backupDB);
        if (rc != SQLITE_OK && rc != SQLITE_DONE)
            throw sql::Exception(rc);
    }
}

void Database::saveDatabase(const char database[])
{
    sql::DB dstDB(database, 0);

    // save to dst database
    saveDatabase(dstDB);
}

void Database::saveDatabase(const int rank)
{
    char database[64];
    MASTER_TMP_FILE(database, rank);

    // get the IDs of process commRank
    int start, end;
    split(start, end, rank);

    // CLOG(INFO, "LOGGER_SYS") << "Start ID: " << start;
    // CLOG(INFO, "LOGGER_SYS") << "End ID: " << end;

    vector<const char*> sqls;

    if (_mode == PARTICLE_MODE)
        sqls = { "insert into dst.groups select distinct groups.* from \
                  groups, particles where \
                  (particles.groupID = groups.ID) and \
                  (particles.ID >= ?1) and (particles.ID <= ?2);",
                 "insert into dst.micrographs \
                  select distinct micrographs.* from \
                  micrographs, particles where \
                  particles.micrographID = micrographs.ID and \
                  (particles.ID >= ?1) and (particles.ID <= ?2);",
                 "insert into dst.particles select * from particles \
                  where (ID >= ?1) and (ID <= ?2);" };
    else if (_mode == MICROGRAPH_MODE)
        sqls = { "insert into dst.micrographs select * from micrographs \
                  where (ID >= ?1) and (ID <= ?2);",
                 "insert into dst.particles select particles.* from \
                  micrographs, particles where \
                  particles.micrographID = micrographs.ID and \
                  (micrographs.ID >= ?1) and (micrographs.ID <= ?2);" };

    _db.exec(("attach database '" + string(database) + "' as dst;").c_str());

    try
    {
        _db.beginTransaction();

        for (const char* sql : sqls)
        {
            sql::Statement stmt(sql, -1, _db);
            stmt.bind_int(1, start);
            stmt.bind_int(2, end);
            stmt.step();
        }

        _db.endTransaction();
    }
    catch (...)
    {
        _db.rollbackTransaction();

        _db.exec("detach database dst;");

        CLOG(FATAL, "LOGGER_SYS") << "Unable to Save Databases for Each Process";
    }

    _db.exec("detach database dst;");
}

void Database::createTables()
{
    createTableGroups();
    createTableMicrographs();
    createTableParticles();
}

void Database::createTableGroups()
{
    _db.exec("create table groups(ID integer primary key, Name text);");

    const char* sql = "insert into groups values (?, ?);";
    _stmtAppendGroup = sql::Statement(sql, strlen(sql), _db);
}

void Database::createTableMicrographs()
{
    _db.exec("create table micrographs(ID integer primary key, \
                                                   Name text, \
                                                   Voltage real not null, \
                                                   DefocusU real not null, \
                                                   DefocusV real not null, \
                                                   DefocusAngle real not null, \
                                                   CS real not null);");

    const char* sql = "insert into micrographs \
                        values (?, ?, ?, ?, ?, ?, ?)";
    _stmtAppendMicrograph = sql::Statement(sql, strlen(sql), _db);
}

void Database::createTableParticles()
{
    _db.exec("create table particles( \
                                  ID integer primary key, \
                                  Name text, \
                                  GroupID integer not null, \
                                  MicrographID integer not null);");

    const char* sql = "insert into particles (Name, GroupID, MicrographID) \
                        values (?, ?, ?)";
    _stmtAppendParticle = sql::Statement(sql, strlen(sql), _db);
}

void Database::appendGroup(const char name[],
                           const int id)
{
    _stmtAppendGroup.reset();
    if (id != -1)
        _stmtAppendGroup.bind_int(1, id);
    else
        _stmtAppendGroup.bind_null(1);

    _stmtAppendGroup.bind_text(2, name, strlen(name), true);
    _stmtAppendGroup.step();
    _stmtAppendGroup.reset();
}

void Database::appendMicrograph(const char name[],
                                const double voltage,
                                const double defocusU,
                                const double defocusV,
                                const double defocusAngle,
                                const double CS,
                                const int id)
{
    _stmtAppendMicrograph.reset();
    if (id != -1)
        _stmtAppendMicrograph.bind_int(1, id);
    else
        _stmtAppendMicrograph.bind_null(1);

    _stmtAppendMicrograph.bind_text(2, name, strlen(name), true);

    _stmtAppendMicrograph.bind_double(3, voltage);

    _stmtAppendMicrograph.bind_double(4, defocusU);

    _stmtAppendMicrograph.bind_double(5, defocusV);

    _stmtAppendMicrograph.bind_double(6, defocusAngle);

    _stmtAppendMicrograph.bind_double(7, CS);

    _stmtAppendMicrograph.step();

    _stmtAppendMicrograph.reset();
}

void Database::appendParticle(const char name[],
                              const int groupID,
                              const int micrographID)
{
    _stmtAppendParticle.reset();
    _stmtAppendParticle.bind_text(1, name, strlen(name), true);
    _stmtAppendParticle.bind_int(2, groupID);
    _stmtAppendParticle.bind_int(3, micrographID);
    _stmtAppendParticle.step();
    _stmtAppendParticle.reset();
}

int Database::nParticle() const
{
    sql::Statement count("select count(*) from particles", -1, _db);
    if (count.step())
        return count.get_int(0);
    return 0;
}

int Database::nMicrograph() const
{
    sql::Statement count("select count(*) from micrographs", -1, _db);
    if (count.step())
        return count.get_int(0);
    return 0;
}

int Database::nGroup() const
{
    sql::Statement count("select count(*) from groups", -1, _db);
    if (count.step())
        return count.get_int(0);
    return 0;
}

void Database::update(const char database[],
                      const Table table)
{
        string sql = "attach database '" + string(database) + "' as src";
        _db.exec(sql.c_str());
    try
    {
        _db.beginTransaction();

        switch (table) {
        case Groups:
            _db.exec("replace into particles select * from src.particles");
            break;

        case Micrographs:
            _db.exec("replace into micrographs select * from src.micrographs");
            break;

        case Particles:
            _db.exec("replace into particles select * from src.particles");
            break;
        }

        _db.endTransaction();
    }
    catch (...)
    {
        _db.rollbackTransaction();
        _db.exec("detach database src");
        throw;
    }
        _db.exec("detach database src");
}

void Database::prepareTmpFile()
{
    if (_commRank == 0)
        masterPrepareTmpFile();
    else
        slavePrepareTmpFile();
}

void Database::gather()
{
    if (_commRank == 0)
        for (int i = 1; i < _commSize; i++)
            masterReceive(i);
    else
        slaveSend();
}

void Database::scatter()
{
    if (_commRank == 0)
    {
        CLOG(INFO, "LOGGER_SYS") << "Total Number of Particles: " << nParticle();
        CLOG(INFO, "LOGGER_SYS") << "Total Number of Micrographs: " << nMicrograph();
        CLOG(INFO, "LOGGER_SYS") << "Total Number of Groups: " << nGroup();

        for (int i = 1; i < _commSize; i++)
        {
            saveDatabase(i);
            masterSend(i);
        }
    }
    else
        slaveReceive();
}

void Database::split(int& start,
                     int& end,
                     const int commRank) const
{
    int size;
    if (_mode == PARTICLE_MODE)
        size = nParticle();
    else if (_mode == MICROGRAPH_MODE)
        size = nMicrograph();
    else
        __builtin_unreachable();

    int piece = size / (_commSize - 1);

    if (commRank <= size % (_commSize - 1)) {
        start = (piece + 1) * (commRank - 1) + 1;
        end = start + (piece + 1) - 1;
    }
    else
    {
        start = piece * (commRank - 1) + size % (_commSize - 1) + 1;
        end = start + piece - 1;
    }
}

void Database::setTempInMemory()
{
    _db.exec("pragma temp_store = memory");
}

void Database::finalizeStatement()
{
    // no-op
}

void Database::masterPrepareTmpFile()
{
    if (_commRank != 0)
        return;

    // open dst database

    char die[256];
    char cast[256];
    char cmd[128];

    sprintf(cmd, "mkdir /tmp/%s", _ID);
    system(cmd);
    sprintf(cmd, "mkdir /tmp/%s/m", _ID);
    system(cmd);
    sprintf(cmd, "mkdir /tmp/%s/s", _ID);
    system(cmd);

    MASTER_TMP_FILE(die, 0);
    sprintf(cmd, "rm -f %s", die);
    system(cmd);
    sprintf(cmd, "touch %s", die);
    system(cmd);

    string sql;
    // create a database struct for each node
    sql = "attach database '" + string(die) + "' as dst";
    _db.exec(sql.c_str());
    try
    {
        _db.beginTransaction();

        _db.exec("create table dst.groups as select * from groups where 1=0");

        _db.exec("create table dst.micrographs as select * from micrographs where 1=0");

        _db.exec("create table dst.particles as select * from particles where 1=0");

        _db.endTransaction();
    }
    catch (...)
    {
        _db.rollbackTransaction();
        _db.exec("detach database dst");

        throw;
    }
        _db.exec("detach database dst");


    for (int i = 1; i < _commSize; i++) {
        MASTER_TMP_FILE(cast, i);
        system(cmd);
        sprintf(cmd, "cp %s %s", die, cast);
        system(cmd);
    }
}

void Database::slavePrepareTmpFile()
{
    if (_commRank == 0) return;

    char cmd[128];
    sprintf(cmd, "mkdir /tmp/%s", _ID);
    system(cmd); 
    sprintf(cmd, "mkdir /tmp/%s/s", _ID);
    system(cmd); 
}

void Database::masterReceive(const int rank)
{
    if (_commRank != 0)
        return;

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
    if (_commRank == 0)
        return;

    MPI_Status status;

    char database[128];
    SLAVE_TMP_FILE(database);

    char* buf = new char[MAX_LENGTH];

    MPI_Recv(buf, MAX_LENGTH, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);

    int len;
    MPI_Get_count(&status, MPI_BYTE, &len);

    WRITE_FILE(database, buf, len);

    delete[] buf;

    openDatabase(database);
}

void Database::masterSend(const int rank)
{
    if (_commRank != 0)
        return;

    char database[128];
    MASTER_TMP_FILE(database, rank);

    char* buf = new char[MAX_LENGTH];

    int len = READ_FILE(database, buf);
    MPI_Ssend(buf, len, MPI_BYTE, rank, 0, MPI_COMM_WORLD);

    delete[] buf;
}

void Database::slaveSend()
{
    if (_commRank == 0)
        return;

    char database[128];
    SLAVE_TMP_FILE(database);

    char* buf = new char[MAX_LENGTH];

    int len = READ_FILE(database, buf);
    MPI_Ssend(buf, len, MPI_BYTE, 0, 0, MPI_COMM_WORLD);

    delete[] buf;
}
