#include "Preprocess.h"
#include "Experiment.h"

/* TODO list: 
  3  invert is not implemented yet
  4  OMP #pragma is not written yet


  X1  normalise() : how to fill the parameter?
  X2  GET_PARTICLE_INFO:  how to write Lambda function and where to get x, y?
*/


Preprocess::Preprocess()
{

}

Preprocess::Preprocess(PREPROCESS_PARA& para,
                       Experiment* exp)
{
    _para = para;
    _exp = exp;
}


PREPROCESS_PARA& Preprocess::getPara()  
{
    return  _para;  // protected?
}

void Preprocess::setPara(PREPROCESS_PARA& para)  
{
    _para = para;    
}

/****************************************************************** 
  extractParticles(): extract particles from specified micrographID
  
  
*******************************************************************/
void Preprocess::extractParticles(const int micrographID)
{
    // x, y
    // micrograph
    // _exp
    // save

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

    // get all particleID in micrographID;
    vector<int> particleIDs;
    _exp->particleIDsMicrograph(particleIDs, micrographID);
    if (particleIDs.size() == 0)
    {
        return ;
    }
 
    ImageFile micrographFile(micName, "rb");
    Image micrograph;
    
    //  read micrograph image 
    micrographFile.readMetaDataMRC();
    micrographFile.readImage(micrograph, 0, "MRC");    
//    micrographFile.display();

    ImageFile particleFile;
    Image particle(_para.nCol, _para.nRow, RL_SPACE);

   
    //printf("[x1]particleIDs.size() = %d nRow=%d  col=%d \n", particleIDs.size() , _para.nCol, _para.nRow );

    //#pragma omp parallel for
    for (int i = 0; i < particleIDs.size(); i++)
    {
        int xOff, yOff;

        // extractPartcilesInMicrograph(micrographImage, particleIDs[i], particleImage  );
        getParticleXOffYOff(xOff, yOff, particleName, particleIDs[i]);
    
        //rintf("xOff =%d  yOff = %d  col=%d  row=%d \n",  xOff, yOff, _para.nCol, _para.nRow );    

        xOff = xOff - micrographFile.nCol() /2 + _para.nCol /2;
        yOff = yOff - micrographFile.nRow() /2 + _para.nRow /2;
        
        //printf("[x1] xOff=%d  yOff=%d \n", xOff, yOff);
        extract(particle, micrograph, xOff, yOff);
        //printf("[x2]");
        
        if (_para.doNormalise)
        {
            normalise(particle,
                      _para.wDust,
                      _para.bDust,
                      _para.r);  
        }
        
        //printf("[x3]");
        if (_para.doInvertConstrast )
        {
            //invertContrast(particle);
            NEG_RL(particle);
        }
        //printf("[x4] particleName =%s \n", particleName);
        #if 1   
        //ImageFile particleFile("/dev/shm/xx", "wb");
        particleFile.readMetaData(particle );        
        particleFile.writeImage(particleName, particle);
        //printf("[x5]");
        #endif
        
    }    
}



/****************************************************************** 
  getMicrographIDs(): get micrograph IDs and saved in  dst vectors   
*******************************************************************/
void Preprocess::getMicrographIDs(vector<int>& dst )
{
    dst.clear();
    //GET_MIC_ID(dst,  start, end);  // ???

    char sql[128]; 
    sprintf(sql, "select ID from micrographs;"); 

    _exp->execute(sql,
                  SQLITE3_CALLBACK
                  {
                      ((vector<int>*)data)
                      ->push_back(atoi(values[0]));
                      return 0;
                  },
                  &dst);      
}


/****************************************************************** 
  getMicrographName(): get micrograph names for specific micrograph ID  
*******************************************************************/
void Preprocess::getMicrographName(char micrographName[], 
                                   const int micrographID
                                   )
{
    
    //GET_MIC_NAME(micName, micrographID);  // ???
    char sql[128]; 

    if (micrographName == NULL)
    {
        REPORT_ERROR("Micrograph name pointer is NULL!\n");
        return ;
    }

    sprintf(sql, "select Name from micrographs where ID = %d;", micrographID); 

    _exp->execute(sql,
                  SQLITE3_CALLBACK
                  {
                      sprintf( (char *)data , "%s", values[0]); 
                      return 0;
                  },
                  micrographName);
    
}


/****************************************************************** 
  getParticleXOffYOff(): get particle XOff,YOff, particle names for
                         specific particle ID  
*******************************************************************/
void Preprocess::getParticleXOffYOff(int& xOff,
                                     int& yOff,
                                     char particleName[], 
                                     const int particleID                                     
                                     )
{
    // GET_PARTICLE_INFO( micrographID, particleID, x, y);  

    PARTICLE_INFO info; 
    char sql[128]; 
    sprintf(sql, 
            "select XOff, YOff, Name from particles "
            "where ID = %d;", 
            particleID); 

    

    _exp->execute(sql,
                  SQLITE3_CALLBACK
                  {
                      ((PARTICLE_INFO*)data)->x = atoi(values[0]);  
                      ((PARTICLE_INFO*)data)->y = atoi(values[1]);  
                      sprintf(  ((PARTICLE_INFO*)data)->particleName,"%s", values[2]);  
                      return 0;
                  },
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

    
    getMicrographIDs(_micrographIDs);
    
    //#pragma omp parallel for
    printf("_micrographIDs.size()=%d \n ", _micrographIDs.size());

    for (int i = 0; i < _micrographIDs.size(); i++)
    {
    	  //printf(" _micrographIDs[%d]= %d \n", i, _micrographIDs[i]);
        extractParticles(_micrographIDs[i]);
    }
}


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




/****************************************************************** 
  removeOutboundParticles(): extract particles from specified micrographID
  
  
*******************************************************************/
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
