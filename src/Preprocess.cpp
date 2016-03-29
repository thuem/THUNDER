/*******************************************************************************
 * Author: Mingxu Hu, Bing Li
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Preprocess.h"

Preprocess::Preprocess() {}

Preprocess::Preprocess(const PreprocessPara& para)
{
    _para = para;
    _exp.openDatabase(_para.db);
}

PreprocessPara& Preprocess::getPara()  
{
    return _para;
}

void Preprocess::setPara(const PreprocessPara& para)  
{
    _para = para;    
}

void Preprocess::run()
{
    _exp.openDatabase(_para.db);

    _exp.bcastID();

    _exp.setMPIEnv(_commSize, _commRank, _hemi);

    _exp.prepareTmpFile();

    _exp.setMode(MICROGRAPH_MODE);

    _exp.scatter();

    if (_commRank != 0)
    {
        getMicIDs(_micIDs);

        for (int i = 0; i < _micIDs.size(); i++)
        {
            cout << _micIDs[i] << endl;
            extractParticles(_micIDs[i]);
        }
    }

    _exp.gather();
}

void Preprocess::getMicIDs(vector<int>& dst)
{
    dst.clear();

    char sql[] = "select ID from micrographs;";

    _exp.execute(sql,
                 SQLITE3_CALLBACK
                 {
                     ((vector<int>*)data)
                     ->push_back(atoi(values[0]));
                     return 0;
                 },
                 &dst); 
}

void Preprocess::getMicName(char micName[], 
                            const int micID)
{
    char sql[SQL_COMMAND_LENGTH]; 

    sprintf(sql, "select Name from micrographs where ID = %d;", micID); 

    _exp.execute(sql,
                 SQLITE3_CALLBACK
                 {
                     sprintf((char*)data, "%s", values[0]); 
                     return 0;
                 },
                 micName);
}

void Preprocess::getParXOffYOff(int& xOff,
                                int& yOff,
                                const int parID)
{
    XY xy;
    char sql[SQL_COMMAND_LENGTH]; 
    sprintf(sql, 
            "select XOff, YOff from particles where ID = %d;", 
            parID); 

    _exp.execute(sql,
                 SQLITE3_CALLBACK
                 {
                     ((XY*)data)->x = atoi(values[0]);  
                     ((XY*)data)->y = atoi(values[1]);  
                     return 0;
                 },
                 &xy);

    xOff = xy.x;
    yOff = xy.y;
}

void Preprocess::extractParticles(const int micID)
{
    char micName[FILE_NAME_LENGTH];
    getMicName(micName, micID);
    string sMicName(micName);

    char parName[FILE_NAME_LENGTH];    

/***
    if ( 0 != ::access(micName, F_OK) )
    {
        char msg[256];
        sprintf(msg, "[Error] micrograph file %s doesn't exists .\n", micName);
        REPORT_ERROR(msg);        
        return ;
    };
***/

    vector<int> parIDs;
    _exp.particleIDsMicrograph(parIDs, micID);

    if (parIDs.size() == 0) return;
 
    Image mic;

    // read in micrograph
    ImageFile micFile(micName, "r");
    micFile.readMetaData();
    micFile.readImage(mic, 0);    

    Image par(_para.nCol, _para.nRow, RL_SPACE);

    ImageFile parFile;

    char sql[SQL_COMMAND_LENGTH]; 

    for (int i = 0; i < parIDs.size(); i++)
    {
        int xOff, yOff;
        getParXOffYOff(xOff, yOff, parIDs[i]);
    
        xOff -= mic.nColRL() / 2 - par.nColRL() / 2;
        yOff -= mic.nRowRL() / 2 - par.nRowRL() / 2;
        
        extract(par, mic, xOff, yOff);
        
        if (_para.doNormalise)
            normalise(par,
                      _para.wDust,
                      _para.bDust,
                      _para.r);  

        if (_para.doInvertConstrast)
            NEG_RL(par);

        // generate parName according to micName and i
        sprintf(parName,
                "%s_%04d.mrc",
                sMicName.substr(0, sMicName.rfind('.') - 1).c_str(),
                i);
        parFile.readMetaData(par);
        parFile.writeImage(parName, par);

        // write parName to database
        sprintf(sql,
                "update particles set Name = \"%s\" where ID = %d;", 
                parName,
                parIDs[i]); 
        _exp.execute(sql, NULL, NULL);
    }
}
