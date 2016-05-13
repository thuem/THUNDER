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

#define DBNAME   "test.db"
#define SHELL_RM "rm "
#define RM_DB    "rm /dev/shm/test.db"
#define MICROGRAPH_PATH "/home/humingxu/Micrographs"
#define STAR_PATH        "/home/humingxu/Star"

using namespace std;

void initPara(PREPROCESS_PARA* para)
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
     strcpy(para->db, "./test.db");
}

void readStar(Experiment& exp, char *micrographFileName, char *starFileName)
{

    FILE *fdStar;
    FILE *fdMicrograph;
    
    int len;

    int  particleNumber;
    int  micrographID;


    char buffer[1024*8];
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


    while ( !feof(fdStar) )
    {
        fscanf(fdStar, "%s\n", buffer);
        if (buffer[0] == '#'  &&  buffer[1] == '5')
        {
           break;
        }
    };  

    sql::Statement stmt("insert into Micrographs "\
                          "( Name, Voltage, DefocusU, DefocusV, DefocusAngle, CA  ) "\
                          "VALUES (?, 1, 2, 3, 4, 5 ); ", -1, exp.expose());
    stmt.bind_text(1, micrographFileName, strlen(micrographFileName), false);
    stmt.step();
    
    stmt = sql::Statement("select ID from Micrographs "\
                         "where Name== ? ;", -1, exp.expose());
    stmt.bind_text(1, micrographFileName, strlen(micrographFileName), false);

    while (stmt.step())
        micrographID = stmt.get_int(0);

        stmt = sql::Statement("insert into particles "\
                              "(XOff, YOff, Name ,GroupID ,MicrographID  ) "\
                              "VALUES (?, ?, ? ,0 , ? ); ", -1, exp.expose());
    printf(" micrographID = %d \n", micrographID);

    particleNumber = 0;
    while ( !feof(fdStar) )
    {          
         
        fscanf(fdStar, "%lf %lf %lf  %d  %lf\n", &CooridinateX, &CooridinateY, 
               &AnglePsi, &ClassNumber , &AutopickFigureOfMerit);
                   
        sprintf(particleName, "%s", micrographFileName);
        len = strlen( particleName );
        if ( strcmp( particleName+ len -4, ".mrc") == 0)
        {
          len -= 4;
        };        
        sprintf( particleName + len, "_%d.mrc", particleNumber);

        printf(" particleName=%s \n", particleName);

	ImageFile imf(micrographFileName, "rb");
	imf.readMetaData();
        if ((CooridinateX >= 0) &&
            (CooridinateY >= 0) &&
            (CooridinateX < imf.nCol() - 300) &&
            (CooridinateY < imf.nRow() - 300))
        {
            stmt.bind_double(1, CooridinateX);
            stmt.bind_double(2, CooridinateY);
            stmt.bind_text(3, particleName, strlen(particleName), false);
            stmt.bind_int(4, micrographID);
            stmt.step();
            stmt.reset();
            particleNumber++;
        }
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
    exp.createTableGroups();
    exp.addColumnXOff();
    exp.addColumnYOff();
  
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

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{   
    loggerInit();

/***
    system("cp test.db  /dev/shm/test.db");

    
    printf("%s\n", RM_DB);

    system(RM_DB);

    Experiment exp(DBNAME);

    createDB(exp);
***/

    /***
    exp.createTableParticles();
    exp.addColumnXOff();
    exp.addColumnYOff();
    exp.addColumnParticleName();

    exp.createTableMicrographs();
    */


    


     /***
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
    ***/

    MPI_Init(&argc, &argv);

    PreprocessPara para;    
    initPara(&para);

    Preprocess preprocess(para);
    preprocess.setMPIEnv();

    try
    {
    	preprocess.run();
    }
    catch (Error& err)
    {
        cout << err;
    }

    MPI_Finalize();

    return 0;
}
