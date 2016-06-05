/*******************************************************************************
 * Author: Mingxu Hu, Bing Li, Siyuan Ren
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

PreprocessPara& Preprocess::para()  
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

        for (size_t i = 0; i < _micIDs.size(); i++)
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

    sql::Statement stmt("select ID from micrographs;", -1, _exp.expose());
    while (stmt.step())
        dst.push_back(stmt.get_int(0));
}

std::string Preprocess::getMicName(const int micID)
{
    sql::Statement stmt("select Name from micrographs where ID = ?;", -1, _exp.expose());
    stmt.bind_int(1, micID);

    if (stmt.step())
        return stmt.get_text(0);
    else
        CLOG(FATAL, "LOGGER_SYS") << "No Micrograph Name";
    __builtin_unreachable();
}

void Preprocess::getParXOffYOff(int& xOff,
                                int& yOff,
                                const int parID)
{
    sql::Statement stmt("select XOff, YOff from particles where ID = ?", -1, _exp.expose());
    stmt.bind_int(1, parID);
  
    if (stmt.step())
    {
        xOff = stmt.get_int(0);
        yOff = stmt.get_int(1);
    }
    else
        CLOG(FATAL, "LOGGER_SYS") << "No xOff, yOff";
}

void Preprocess::extractParticles(const int micID)
{
    string sMicName = getMicName(micID);

    char parName[FILE_NAME_LENGTH];    

    vector<int> parIDs;
    _exp.particleIDsMicrograph(parIDs, micID);

    if (parIDs.size() == 0) return;
 
    Image mic;

    // read in micrograph
    ImageFile micFile(sMicName.c_str(), "r");
    micFile.readMetaData();
    micFile.readImage(mic, 0);    

    Image par(_para.nCol, _para.nRow, RL_SPACE);

    ImageFile parFile;

    sql::Statement updator("update particles set Name = ? where ID = ?", -1, _exp.expose());

    for (size_t i = 0; i < parIDs.size(); i++)
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
        snprintf(parName, sizeof(parName),
                "%s_%04zu.mrc",
                sMicName.substr(0, sMicName.rfind('.') - 1).c_str(),
                i);
        parFile.readMetaData(par);
        parFile.writeImage(parName, par);

        updator.bind_text(1, parName, strlen(parName), false);
        updator.bind_int(2, parIDs[i]);
        updator.step();
        updator.reset();
    }
}
