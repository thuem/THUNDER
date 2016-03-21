/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef Experiment_H
#define Experiment_H

#include <vector>
#include <functional>

#include "Database.h"

using namespace std;

#define ADD_COLUMN(TABLE, COLUMN, ATTR) \
{ \
    char sql[128]; \
    sprintf(sql, \
            "alter table %s add column %s %s;", \
            #TABLE, \
            #COLUMN, \
            #ATTR); \
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, \
                                      sql, \
                                      NULL, NULL, NULL)); \
}


#define GET_ID(dst, TABLE, COLUMN, value)\
[this](vector<int>& _dst, int _value) \
{ \
    char sql[128]; \
    sprintf(sql, \
            "select ID from %s where %s = %d;", \
            #TABLE, \
            #COLUMN, \
            _value); \
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, \
                                      sql, \
                                      SQLITE3_CALLBACK \
                                      { \
                                          ((vector<int>*)data) \
                                          ->push_back(atoi(values[0])); \
                                          return 0; \
                                      }, \
                                      &_dst, \
                                      NULL)); \
}(dst, value)

class Experiment : public Database
{
    private:

    public:

        Experiment();

        Experiment(const char database[]);

        void execute(const char sql[],
                     int(*func)(void*, int, char**, char**),
                     void* data);

        void addColumnXOff();

        void addColumnYOff();

        void addColumnParticleName();

        void particleIDsMicrograph(vector<int>& dst,
                                   const int micrographID);
        /* return IDs of particles belonging to a certain micrograph */

        void particleIDsGroup(vector<int>& dst,
                              const int groupID);
        /* return IDs of particles belonging to a certain group*/

        /***
        void appendParticle(const int groupID,
                            const int micrographID,
                            const int XOff,
                            const int YOff);
                            ***/
};

#endif // Experiment_H
