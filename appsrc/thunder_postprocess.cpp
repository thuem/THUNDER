//This header file is add by huabin
#include "huabin.h"
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

    TSFFTW_init_threads();

    Postprocess pp(argv[1],
                   argv[2],
                   argv[3],
                   atof(argv[4]));

    pp.run();

    TSFFTW_cleanup_threads();

    return 0;
}
