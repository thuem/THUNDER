/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Database.h"

#define N 1000

using namespace std;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    Database db;

    db.createTables();

    db.appendMicrograph("", 0, 0, 0, 0, 0);
    db.appendGroup("");

    for (int i = 0; i < N; i++)
        db.appendParticle("", 1, 1);

    db.setMPIEnv();

    db.bcastID();
    db.prepareTmpFile();

    /***
    db.scatter();
    db.gather();   
    ***/

    MPI_Finalize();
}
