/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Experiment.h"

Experiment::Experiment() : Database() {}

Experiment::Experiment(const char database[]) : Database(database) {}

void Experiment::addColumnXOff()
{
    ADD_COLUMN(particles, XOff, integer);
}

void Experiment::addColumnYOff()
{
    ADD_COLUMN(particles, YOff, integer);
}

void Experiment::addColumnParticleName()
{
    ADD_COLUMN(particles, Name, text);
}

void Experiment::particleIDsMicrograph(vector<int>& dst,
                                       const int micrographID)
{
    dst.clear();

    GET_ID(dst, particles, MicrographID, micrographID);
    /***
    char sql[128];
    sprintf(sql,
            "select ID from particles where MicrographID = %d;",
            micrographID);

    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db,
                                      sql,
                                      SQLITE3_CALLBACK
                                      {
                                          ((vector<int>*)data)->push_back(atoi(values[0]));
                                          return 0;
                                      },
                                      &dst,
                                      NULL));
                                      ***/
} 
