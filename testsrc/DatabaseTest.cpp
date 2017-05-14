/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Database.h"

#define TEST_N_PARTICLE

//#define TEST_N_PARTICLE_RANK

//#define TEST_START_END

//#define TEST_N_GROUP

//#define TEST_OFFSET

//#define TEST_GROUP_ID

//#define TEST_PATH

//#define TEST_CTF

//#define TEST_CLS

//#define TEST_QUAT

//#define TEST_STD_R

//#define TEST_TRAN

//#define TEST_D

#define TEST_STD_D

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

#ifdef TEST_START_END
    std::cout << "Start = " << db.start() << std::endl;
    std::cout << "End = " << db.end() << std::endl;
#endif

#ifdef TEST_N_GROUP
    std::cout << "Number of Groups : "
              << db.nGroup()
              << std::endl;
#endif

    std::cout << "Indexing" << std::endl;

    db.index();

    std::cout << "Shuffling" << std::endl;

    db.shuffle();

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

    for (int i = db.nParticle() - 1; i >= db.nParticle() - 10; i--)
        std::cout << db.path(i) << std::endl;
#endif

#ifdef TEST_CTF
    std::cout << "CTF " << std::endl;
    double voltage, defocusU, defocusV, defocusTheta, Cs, amplitudeContrast, phaseShift;

    for (int i = 0; i < 10; i++)
    {
        db.ctf(voltage,
               defocusU,
               defocusV,
               defocusTheta,
               Cs,
               amplitudeContrast,
               phaseShift,
               i);

        std::cout << "Voltage = " << voltage << std::endl
                  << "defocusU = " << defocusU << std::endl
                  << "defocusV = " << defocusV << std::endl
                  << "defocusTheta = " << defocusTheta << std::endl
                  << "Cs = " << Cs << std::endl
                  << "amplitudeContrast = " << amplitudeContrast << std::endl
                  << "phaseShift = " << phaseShift << std::endl;
    }
#endif

#ifdef TEST_CLS

    std::cout << "cls " << std::endl;

    for (int i = 0; i < 10; i++)
        std::cout << db.cls(i) << std::endl;

#endif

#ifdef TEST_QUAT

    std::cout << "quat " << std::endl;

    for (int i = 0; i < 10; i++)
        std::cout << db.quat(i) << std::endl << std::endl;

#endif

#ifdef TEST_STD_R

    std::cout << "stdR " << std::endl;

    for (int i = 0; i < 10; i++)
        std::cout << db.stdR(i) << std::endl;

#endif

#ifdef TEST_TRAN

    std::cout << "tran " << std::endl;

    for (int i = 0; i < 10; i++)
        std::cout << db.tran(i) << std::endl << std::endl;

#endif

#ifdef TEST_D

    std::cout << "d " << std::endl;

    for (int i = 0; i < 10; i++)
        std::cout << db.d(i) << std::endl;

#endif

#ifdef TEST_STD_D

    std::cout << "stdD " << std::endl;

    for (int i = 0; i < 10; i++)
        std::cout << db.stdD(i) << std::endl;

#endif

    MPI_Finalize();
}
