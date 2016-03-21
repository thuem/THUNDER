#include "Preprocess.h"

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
    char micName[FILE_NAME_LENGTH];
    char parName[FILE_NAME_LENGTH];    

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
    micrographFile.readMetaDataMRC();
    micrographFile.readImage(micrograph, 0, "MRC");    
//    micrographFile.display();

    ImageFile particleFile;
    Image particle(_para.nCol, _para.nRow, RL_SPACE);

    #pragma omp parallel for
    for (int i = 0; i < particleIDs.size(); i++)
    {
        int xOff, yOff;

        getParticleXOffYOff(xOff, yOff, parIDs[i]);
    
        xOff = xOff - micrographFile.nCol() / 2 + _para.nCol / 2;
        yOff = yOff - micrographFile.nRow() / 2 + _para.nRow / 2;
        
        //printf("[x1] xOff=%d  yOff=%d \n", xOff, yOff);
        extract(particle, micrograph, xOff, yOff);
        //printf("[x2]");
        
        if (_para.doNormalise)
            normalise(particle,
                      _para.wDust,
                      _para.bDust,
                      _para.r);  

        if (_para.doInvertConstrast)
            NEG_RL(particle);

        // TODO: generate pareName
        particleFile.readMetaData(particle);        
        particleFile.writeImage(parName, particle);
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

<<<<<<< HEAD
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
=======
                  &info);
    
    xOff = info.x;
    yOff = info.y;
    sprintf(particleName, "%s", info.particleName); 
    
}

/****************************************************************** 
  getParticleXOffYOff(): get particle XOff,YOff, particle names for
                         specific particle ID  
*******************************************************************/
void Preprocess::run()
{   
    // get all micrographID;
    //vector<int> micrographIDs;e

    dbPreprocess();

    
>>>>>>> 9e15aff2a2d70bac731bffe997fbe5785fd390f2
    getMicrographIDs(_micrographIDs);
    
    //#pragma omp parallel for
    printf("_micrographIDs.size()=%d \n ", _micrographIDs.size());

    for (int i = 0; i < _micrographIDs.size(); i++)
    {
    	  //printf(" _micrographIDs[%d]= %d \n", i, _micrographIDs[i]);
        extractParticles(_micrographIDs[i]);
    }
}
<<<<<<< HEAD
=======


void Preprocess::dbPreprocess()
{

    getMicrographIDs(_micrographIDs);
    
    //#pragma omp parallel for
    for (int i = 0; i < _micrographIDs.size(); i++)
    {
        //printf(" _micrographIDs[%d]= %d \n", i, _micrographIDs[i]);
        removeOutboundParticles(_micrographIDs[i]);
    }
}

void Preprocess::removeOutboundParticles(const int micrographID)
{
    // x, y
    // micrograph
    // _exp
    // save
    int  XL, XR, YT, YB;
    char sql[1024];
    char micName[FILE_NAME_LENGTH];
    char particleName[FILE_NAME_LENGTH];    

    getMicrographName(micName, micrographID);
    if ( 0 != ::access(micName, F_OK) )
    {
        char msg[256];
        sprintf(msg, "[Error] micrograph file %s doesn't exists .\n", micName);
        REPORT_ERROR(msg);        
        return ;
    };


    ImageFile micrographFile(micName, "rb");
    Image micrograph;
    
    //  read micrograph image 
    micrographFile.readMetaDataMRC();         

    ImageFile particleFile;
    Image particle(_para.nCol, _para.nRow, RL_SPACE);

    // particles coordinates indicates the particle left-top corner index
    XL = 0;
    XR = micrographFile.nCol()  - _para.nCol -1 ;

    YT = 0;
    YB = micrographFile.nRow()  - _para.nRow -1 ;

    sprintf(sql, 
            "delete  from particles "
            "where ID = %d and abs(XOff) < %d or  abs(XOff) > %d "
            "or abs(YOff) < %d or  abs(YOff) > %d   ;", 
            micrographID, XL, XR, YT , YB ); 
    //printf( "%s\n", sql );

    _exp->execute(sql,
                  NULL,
                  NULL);  
   
}
