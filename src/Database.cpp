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
    // TODO
}

int Database::nGroup()
{
    // TODO
}

int Database::nParticleRank()
{
    // TODO
}

void Database::assign()
{
    IF_MASTER return;

    split(_start, _end, _commRank);
}

void Database::split(int& start,
                     int& end,
                     const int commRank)
{
    int size = nParticle();

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
