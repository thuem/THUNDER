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




#define GET_MIC_ID(dst, start, end )\
[this](vector<int>& _dst, int _start, int _end) \
{ \
    char sql[128]; \
    sprintf(sql, \
            "select ID from micrographs  where  %d <= ID  and  ID <= %d ;", \
            _start, \
            _end \
            ); \
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
}(dst,start, end)

typedef struct _XY {int x; int y;} XY;
#define GET_PARTICLE_INFO(micrographID, particleID, x, y) \
[this](int _micographID, int _particleID, int& _x, int& _y) \
{ \
    XY xy; \
    char sql[128]; \
    sprintf(sql, \
            "select XOff, YOFF from particles where MicrographID = %d and ID = %d;", \
            _micographID, \
            _particleID); \
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, \
                                      sql, \
                                      SQLITE3_CALLBACK \
                                      { \
                                            ((XY*)data)->x = atoi(values[0]);  \
                                            ((XY*)data)->y = atoi(values[1]);  \
                                        return 0; \
                                      }, \
                                      &xy, \
                                      NULL)); \
}(micrographID, particleID, x, y)





#if 0
#define GET_PARTICLE_INFO(micrographID, particleID, x, y) \
[this](int _micId, int  _parId,  int & _x, int & _y) \
{ \
    struct   XY _dst; \
    char sql[128]; \
    sprintf(sql, \
            "select x, y from micrographs  where  MicrographID = %d  and ID = %d ;", \
            _micId, \
            _parId \
            ); \
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, \
                                      sql, \
                                      SQLITE3_CALLBACK \
                                      { \
                                            ((struct XY *)data)->x=     atoi(values[0];  \
                                            ((struct XY *)data)->y=     atoi(values[1];  \
                                            return 0; \
                                      }, \
                                      &_dst, \
                                      NULL)); \
    x= _dst.x,  y=_dst.y;  \
}(micId, parId, x, y)
#endif 



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







#define GET_MIC_NAME(dst, micId )\
[this](char * _dst, int _micId) \
{ \
    char sql[128]; \
    sprintf(sql, \
            "select Name from micrographs  where  ID  = %d  ;", \
            _micId \
            ); \
    SQLITE3_HANDLE_ERROR(sqlite3_exec(_db, \
                                      sql, \
                                      SQLITE3_CALLBACK \
                                      { \
                                          sprintf( (char *)data , "%s",values[0]); \
                                          return 0; \
                                      }, \
                                      &_dst, \
                                      NULL)); \
}(dst, micId)








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

        void getMicrographIDs(vector<int>& dst ,
                              const int start,
                              const int end);

        void getMicrographName(char micName[], 
                               const int micrographID );

        void getParticleInfo(const int micrographID ,
                             const int particleID ,
                             int& x,
                             int& y);

        /***
        void appendParticle(const int groupID,
                            const int micrographID,
                            const int XOff,
                            const int YOff);
                            ***/
};

#endif // Experiment_H
