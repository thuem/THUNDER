/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Volume.h"

#include "Parallel.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    MPI_Init(&argc, &argv);

    Parallel par;
    par.setMPIEnv();
    display(par);

    Volume vol(760, 760, 760, FT_SPACE);
    SET_0_FT(vol);

    for (int i = 0; i < atoi(argv[1]); i++)
    {
        if (par.commRank() != MASTER_ID)
        {
            if (par.commRank() == HEMI_A_LEAD)
                CLOG(INFO, "LOGGER_SYS") << "HEMI_A: Round " << i;
            if (par.commRank() == HEMI_B_LEAD)
                CLOG(INFO, "LOGGER_SYS") << "HEMI_B: Round " << i;

            MPI_Allreduce_Large(MPI_IN_PLACE,
                                &vol[0],
                                vol.sizeFT(),
                                MPI_DOUBLE_COMPLEX,
                                MPI_SUM,
                                par.hemi());
        }
    }
    
    MPI_Finalize();
}
