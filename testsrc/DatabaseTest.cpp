/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Database.h"

using namespace std;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    Database db;

    db.createTables();

    db.setMPIEnv();

    db.bcastID();
    db.prepareTmpFile();

    db.setMode(MICROGRAPH_MODE);

    db.scatter();
    db.gather();   

    MPI_Finalize();
}
