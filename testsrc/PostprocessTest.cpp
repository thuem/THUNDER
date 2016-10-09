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
    loggerInit();

    Postprocess pp(argv[1], argv[2]);

    return 0;
}
