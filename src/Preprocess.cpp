#include "Preprocess.h"

/* TODO list: 
  3  invert is not implemented yet
  4  OMP #pragma is not written yet


  X1  normalise() : how to fill the parameter?
  X2  GET_PARTICLE_INFO:  how to write Lambda function and where to get x, y?
*/


Preprocess::Preprocess() {}

Preprocess::Preprocess(const PreprocessPara& para,
                       Experiment* exp)
{
    _para = para;
    _exp = exp;
}

PreprocessPara& Preprocess::getPara()  
{
    return _para;
}

void Preprocess::setPara(const PreprocessPara& para)  
{
    _para = para;    
}

void Preprocess::extractParticles(const int micID)
{
    // x, y
    // micrograph
    // _exp
    // save
    
    char micName[FILE_NAME_LENGTH];
    char particleName[FILE_NAME_LENGTH];    

    getMicrographName(micName, micID);
    if ( 0 != ::access(micName, F_OK) )
    {
        char msg[256];
        sprintf(msg, "[Error] micrograph file %s doesn't exists .\n", micName);
        REPORT_ERROR(msg);        
        return ;
    };

    // get all particleID;
    vector<int> parIDs;
    _exp->particleIDsMicrograph(parIDs, micID);
 
    ImageFile micrographFile(micName, "rb");
    Image micrograph;

    //  read micrograph image 
    micrographFile.readImage(micrograph, 0);    
    
    ImageFile particleFile;
    Image particle(_para.nCol, _para.nRow, RL_SPACE);

    #pragma omp parallel for
    for (int i = 0; i < particleIDs.size(); i++)
    {
        int xOff, yOff;

        // extractPartcilesInMicrograph(micrographImage, particleIDs[i], particleImage  );
        getParticleXOffYOff(xOff, yOff, particleName, particleIDs[i]);

        extract(particle, micrograph, xOff, yOff);

        if (_para.doNormalise)
            normalise(particle,
                      _para.wDust,
                      _para.bDust,
                      _para.r);  

        if (_para.doInvertConstrast)
            NEG_RL(particle);

        particleFile.writeImage(particleName, particle);
    }    
}

void Preprocess::getMicIDs(vector<int>& dst)
{
    dst.clear();

    char sql[] = "select ID from micrographs;";

    _exp->execute(sql,
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
    char sql[128]; 

    sprintf(sql, "select Name from micrographs where ID = %d;", micID); 

    _exp->execute(sql,
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
    char sql[128]; 
    sprintf(sql, 
            "select XOff, YOff from particles where ID = %d;", 
            particleID); 

    _exp->execute(sql,
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

void Preprocess::run()
{   
    getMicrographIDs(_micrographIDs);
    
    //#pragma omp parallel for
    printf("_micrographIDs.size()=%d \n ", _micrographIDs.size());
    for (int i = 0; i < _micrographIDs.size(); i++)
    {
    	//printf(" _micrographIDs[%d]= %d \n", i, _micrographIDs[i]);
        extractParticles(_micrographIDs[i]);
    }
}
