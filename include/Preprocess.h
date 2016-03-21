#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <unistd.h>

#include "Macro.h"

#include "Image.h"
#include "ImageFile.h"
#include "ImageFunctions.h"

#include "Experiment.h"

#include "Parallel.h"

/***
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
***/

/***
typedef struct _PARTICLE_INFO 
{
    int x; 
    int y;
    char particleName[FILE_NAME_LENGTH];
}   PARTICLE_INFO;
***/

typedef struct X_Y
{
    int x;
    int y;
} XY;

/***
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
***/

/***
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
***/

typedef struct PREPROCESS_PARA
{    
    int nCol;    

    int nRow;

    int xOffBias;

    int yOffBias;

    bool doNormalise;

    bool doInvertConstrast;

    double wDust;

    double bDust;

    double r;

} PreprocessPara;

class Preprocess : public Parallel
{
    private:

        PreprocessPara _para;

        Experiment* _exp;

        vector<int> _micIDs;

    public:        

        Preprocess();

        Preprocess(const PreprocessPara& para,
                   Experiment* exp); 

        PreprocessPara& getPara();

        void setPara(const PreprocessPara& para);

        void extractParticles(const int micID);

        void run();

    private:

        void getMicIDs(vector<int>& dst);

        void getMicName(char micName[],
                        const int micID);

        void getParXOffYOff(int& xOff,
                            int& yOff,
                            const int parID);
};

#endif // PREPROCESS_H
