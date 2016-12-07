// clang-format
/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Experiment.h"

Experiment::Experiment()
    : Database()
{
}

Experiment::Experiment(const char database[])
    : Database(database)
{
}

void Experiment::addColumnXOff()
{
    add_column("particles", "XOff", "integer");
}

void Experiment::addColumnYOff()
{
    add_column("particles", "YOff", "integer");
}

void Experiment::particleIDsMicrograph(vector<int>& dst,
                                       const int micrographID)
{
    get_id(dst, "particles", "MicrographID", micrographID);
}

void Experiment::particleIDsGroup(vector<int>& dst,
                                  const int groupID)
{
    get_id(dst, "particles", "GroupID", groupID);
}

void Experiment::add_column(const char* table, const char* column, const char* attr)
{
    char cmd[128];
    snprintf(cmd, sizeof(cmd),
             "alter table %s add column %s %s;", table, column, attr);
    _db.exec(cmd);
}

void Experiment::get_id(vector<int>& dst, const char* table, const char* column, int value)
{
    dst.clear();
    char cmd[128];
    snprintf(cmd, sizeof(cmd),
             "select ID from %s where %s = ?;", table, column);
    sql::Statement stmt(cmd, -1, _db);
    stmt.bind_int(1, value);
    while (stmt.step())
        dst.push_back(stmt.get_int(0));
}
