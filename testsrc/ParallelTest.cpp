/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Parallel.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    Parallel par;
    par.setMPIEnv();
    display(par);
    
    MPI_Finalize();
}
