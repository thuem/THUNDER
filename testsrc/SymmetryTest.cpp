/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <cstdio>
#include <iostream>

#include "Symmetry.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, const char* argv[])
{
    loggerInit();

    try
    {
        Symmetry sym("C15");
        display(sym);
    }
    catch (Error& error)
    {
        std::cout << error;
    }
}
