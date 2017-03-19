/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Database.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    MPI_Init(&argc, &argv);

    Database db("xz.thu");

    db.setMPIEnv();

    std::cout << db.nParticle() << std::endl;

    MPI_Finalize();
}
