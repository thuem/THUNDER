/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include   "Experiment.h"

Experiment::Experiment() : Database() {}

Experiment::Experiment(const char database[]) : Database(database) {}

void Experiment::execute(const char sql[],
                         int(*func)(void*, int, char**, char**),
                         void* data)
{
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, sql, func, data, NULL));
}

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
} 

void Experiment::particleIDsGroup(vector<int>& dst,
                                  const int groupID)
{
    dst.clear();
    GET_ID(dst, particles, GroupID, groupID);
}


void Experiment::getMicrographIDs(vector<int>& dst ,
                                  const int  start,
                                  const int  end)
{
    dst.clear();
    GET_MIC_ID(dst,  start, end);  // ???
    // 
}

void Experiment::getMicrographName(char  *micName, 
                                    int  micrographID )
{
    
    GET_MIC_NAME(micName, micrographID);  // ???
    // 
}


void Experiment::getParticleInfo(const int  micrographID ,
                                 const int  particleID ,
                                 int  & x,
                                 int  & y  )
{
    
    GET_PARTICLE_INFO( micrographID, particleID, x, y);  // ???
    // 
}
