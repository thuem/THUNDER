/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef Experiment_H
#define Experiment_H

#include "Database.h"

#define ADD_COLUMN(TABLE, COLUMN, ATTR) \
{ \
    string sql = string("alter table") \
               + string(#TABLE) \
               + string(" add column ") \
               + string(#COLUMN) \
               + string(" ") \
               + string(#ATTR) \
               + string(";"); \
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, \
                                      sql.c_str(), \
                                      NULL, NULL, NULL)); \
}

class Experiment : public Database
{
    public:

        Experiment();

        Experiment(const char database[]);

        void addColumnXOff();

        void addColumnYOff();

        void addColumnParticleName();

        void addColumnMicrographName();
};

#endif // Experiment_H
