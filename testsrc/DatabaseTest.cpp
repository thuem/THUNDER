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

    try
    {
    	Database db("test.db");

        db.setMPIEnv();

        db.bcastID();
        db.prepareTmpFile();

        db.setMode(MICROGRAPH_MODE);

        db.scatter();
        db.gather();   

    }
    catch (const std::exception& err)
    {
        cerr << err.what() << '\n';
    }

    MPI_Finalize();
}
