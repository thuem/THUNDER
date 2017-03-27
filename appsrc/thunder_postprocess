/*******************************************************************************
 * Author: Ice
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Postprocess.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{   
    loggerInit(argc, argv);

    fftw_init_threads();

    Postprocess pp(argv[1],
                   argv[2],
                   argv[3],
                   atof(argv[4]));

    pp.run();

    fftw_cleanup_threads();

    return 0;
}
