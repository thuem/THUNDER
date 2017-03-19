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
    _db = NULL;
}

Database::Database(const char database[])
{
    openDatabase(database);
}

Database::~Database()
{
    fclose(_db);
}

void Database::openDatabase(const char database[])
{
    _db = fopen(database, "r");

    if (_db == NULL) REPORT_ERROR("FAIL TO OPEN DATABASE");
}

void Database::saveDatabase(const char database[])
{
    // TODO
}

int Database::nParticle()
{
    rewind(_db);

    int result = 0;

    char line[FILE_LINE_LENGTH];

    IF_MASTER
    {
        while (fgets(line, FILE_LINE_LENGTH - 1, _db)) result++;
    }

    MPI_Bcast(&result, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    return result;
}

int Database::nGroup()
{
    // TODO
}

int Database::nParticleRank()
{
    IF_MASTER
    {
        CLOG(WARNING, "LOGGER_SYS") << "NO PARTICLE ASSIGNED TO MASTER PROCESS";

        return 0;
    }

    return (_end - _start + 1);
}

void Database::assign()
{
    split(_start, _end, _commRank);
}

void Database::split(int& start,
                     int& end,
                     const int commRank)
{
    int size = nParticle();

    IF_MASTER return;

    int piece = size / (_commSize - 1);

    if (commRank <= size % (_commSize - 1))
    {
        start = (piece + 1) * (commRank - 1) + 1;
        end = start + (piece + 1) - 1;
    }
    else
    {
        start = piece * (commRank - 1) + size % (_commSize - 1) + 1;
        end = start + piece - 1;
    }
}
