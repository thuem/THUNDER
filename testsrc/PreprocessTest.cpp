/*******************************************************************************
 * Author: Ice
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Preprocess.h"

#define N 8

#define DBNAME   "/dev/shm/test.db"
#define SHELL_RM "rm "
#define RM_DB     "rm /dev/shm/test.db"
#define MICROGRAPH_PATH "/home/icelee/bio/Micrographs"
#define STAR_PATH        "/home/icelee/bio/Star"

using namespace std;

void  initPara(PREPROCESS_PARA *para)
{
     para->nCol = 100;
     para->nRow = 100;
     para->xOffBias = 0;
     para->yOffBias = 0;
     para->doNormalise = true;
     para->doInvertConstrast = true;
     para->wDust = 0.1;
     para->bDust = 0.1;
     para->r = 10;    

}


void readStar(Experiment& exp, char *micrographFileName, char *starFileName)
{

    FILE *fdStar;
    FILE *fdMicrograph;
    
    int len;

    int  particleNumber;
    int  micrographID;


    char buffer[1024*8];
    char sqlStatement[1024];
    char particleName[1024];


    double CooridinateX,CooridinateY, AnglePsi, AutopickFigureOfMerit;
    int    ClassNumber;



    fdMicrograph = fopen( micrographFileName, "r");
    if (NULL == fdMicrograph)
    {
        printf(" Open %s failed.\n ", micrographFileName);
        return;
    }

    fdStar= fopen(starFileName, "rb");
    if (NULL == fdStar)
    {
        printf(" Open %s failed.\n ", starFileName);
        fclose(fdMicrograph);
        return;
    }  


    /* skip header */
    while ( !feof(fdStar) )
    {
        fscanf(fdStar, "%s\n", buffer);
        if (buffer[0] == '#'  &&  buffer[1] == '5')
        {
           break;
        }
    };  

    sprintf(sqlStatement, "insert into Micrographs "\
                          "( Name, Voltage, DefocusU, DefocusV, DefocusAngle, CA  ) "\
                          "VALUES ( \"%s\", 1, 2, 3, 4, 5 ); ",                         
                          micrographFileName);
    
    exp.execute(sqlStatement,
                NULL, 
                NULL);

    sprintf(sqlStatement, "select ID from Micrographs "\
                          "where Name== \"%s\" ;",        
                          micrographFileName);

    exp.execute(sqlStatement,
                  SQLITE3_CALLBACK
                  {
                      *((int*)data) = atoi(values[0]);
                      return 0;
                  },
                  &micrographID); 

    printf(" micrographID = %d \n", micrographID);

    particleNumber = 0;
    while ( !feof(fdStar) )
    {          
         
        fscanf(fdStar, "%lf %lf %lf  %d  %lf\n", &CooridinateX, &CooridinateY, 
               &AnglePsi, &ClassNumber , &AutopickFigureOfMerit);
                   
        sprintf(particleName, "%s/%s", MICROGRAPH_PATH, micrographFileName, particleNumber);
        len = strlen( particleName );
        if ( strcmp( particleName+ len -4, ".mrc") == 0)
        {
          len -= 4;
        };        
        sprintf( particleName + len, "_%d.bmp", particleNumber);

        printf(" particleName=%s \n", particleName);

        sprintf(sqlStatement, "insert into particles "\
                              "(XOff, YOff, Name ,GroupID ,MicrographID  ) "\
                              "VALUES (%f, %f, \"%s\" ,0 ,%d  ); ", 
                              CooridinateX, CooridinateY, 
                              particleName,micrographID);

        //printf("%s\n", sqlStatement);
       
        exp.execute(sqlStatement,
                    NULL, 
                    NULL);
        particleNumber++;
    }

    fclose(fdStar);
    fclose(fdMicrograph);

}


void createDB(Experiment& exp)
{
    char starFileName[1024];
    char micrographFileName[1024];

    int i;

    exp.createTableParticles();
    exp.createTableMicrographs();
    exp.addColumnXOff();
    exp.addColumnYOff();
    exp.addColumnParticleName();
  
    for (i = 211; i <= 326 ; i++)
    {
        sprintf(micrographFileName, "%s/stack_%04d_2x_SumCorr.mrc", MICROGRAPH_PATH , i);
        sprintf(starFileName, "%s/stack_%04d_2x_SumCorr_manual_checked.star", STAR_PATH, i);

        if ( -1 == ::access(micrographFileName, F_OK) )
        {
            printf("Micrograph file-%s not exists.\n ", micrographFileName);
            continue;
        }

        if ( -1 == ::access(starFileName, F_OK) )
        {
            printf("Star file-%s not exists.\n ", starFileName);
            continue;
        }
        printf("*");
        readStar(exp, micrographFileName, starFileName );
        printf(".");
    }

}


int main(int argc, const char* argv[])
{   
      
    printf("%s\n", RM_DB);
    system(RM_DB);

    Experiment exp(DBNAME);
/*
    exp.createTableParticles();
    exp.addColumnXOff();
    exp.addColumnYOff();
    exp.addColumnParticleName();

    exp.createTableMicrographs();

    */


    createDB(exp);


    vector<int> partIDs;

    printf("\nAppending particles FInished. \n");

    exp.particleIDsMicrograph(partIDs, 0);
    #if 0
    for (int i = 0; i < partIDs.size(); i++)
        cout << partIDs[i] << endl;

    exp.particleIDsGroup(partIDs, 0);
    for (int i = 0; i < partIDs.size(); i++)
        cout << partIDs[i] << endl;
    #endif

    PREPROCESS_PARA  para;    
    initPara(&para);

    Preprocess   preprocess(para, &exp);
    preprocess.run();

}
