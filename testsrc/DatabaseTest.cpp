/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Database.h"

//#define TEST_N_PARTICLE

//#define TEST_N_PARTICLE_RANK

//#define TEST_N_GROUP

//#define TEST_OFFSET

//#define TEST_GROUP_ID

#define TEST_PATH

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    MPI_Init(&argc, &argv);

    Database db;

    db.setMPIEnv();

    db.openDatabase("xz.thu");

#ifdef TEST_N_PARTICLE
    std::cout << "Total Number of Particles: "
              << db.nParticle()
              << std::endl;
#endif

    std::cout << "Assigning Particles to Procecesses" << std::endl;

    db.assign();

#ifdef TEST_N_PARTICLE_RANK
    std::cout << "Number of Particles of This Process: "
              << db.nParticleRank()
              << std::endl;
#endif

#ifdef TEST_N_GROUP
    std::cout << "Number of Groups : "
              << db.nGroup()
              << std::endl;
#endif

    std::cout << "Indexing" << std::endl;

    db.index();

#ifdef TEST_OFFSET
    std::cout << "Offset of Each Line: " << std::endl;

    for (int i = 0; i < 10; i++)
        std::cout << db.offset(i) << std::endl;
#endif

#ifdef TEST_GROUP_ID
    std::cout << "GroupID" << std::endl;

    for (int i = 0; i < 10; i++)
        std::cout << db.groupID(i) << std::endl;
#endif

#ifdef TEST_PATH
    std::cout << "Path " << std::endl;

    for (int i = 0; i < 10; i++)
        std::cout << db.path(i) << std::endl;
#endif

    MPI_Finalize();
}
