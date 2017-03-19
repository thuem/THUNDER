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

    Database db;

    db.setMPIEnv();

    db.openDatabase("xz.thu");

    std::cout << "Total Number of Particles: "
              << db.nParticle()
              << std::endl;

    std::cout << "Assigning Particles to Procecesses" << std::endl;

    db.assign();

    std::cout << "Number of Particles of This Process: "
              << db.nParticleRank()
              << std::endl;

    std::cout << "Number of Groups : "
              << db.nGroup()
              << std::endl;

    std::cout << "Indexing" << std::endl;

    db.index();

    std::cout << "Offset of Each Line: " << std::endl;

    for (int i = 0; i < 10; i++)
        std::cout << db.offset(i) << std::endl;

    MPI_Finalize();
}
