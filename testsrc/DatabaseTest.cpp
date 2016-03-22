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

        db.BcastID();
        db.prepareTmpFile();
    }
    catch (Error& err)
    {
        cout << err;
    }

    MPI_Finalize();
}
