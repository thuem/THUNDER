/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Experiment.h"

Experiment::Experiment() : Database()
{
}

Experiment::Experiment(const char database[]) : Database(database)
{
}

void Experiment::addColumnXOff()
{
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db,
                                      "alter table particles add column XOff integer;",
                                      NULL, NULL, NULL));
}

void Experiment::addColumnYOff()
{
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db,
                                      "alter table particles add column YOff integer;",
                                      NULL, NULL, NULL));
}
