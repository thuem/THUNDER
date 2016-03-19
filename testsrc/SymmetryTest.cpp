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

int main(int argc, const char* argv[])
{
    try
    {
        Symmetry sym("C5");
        display(sym);
    }
    catch (Error& error)
    {
        std::cout << error;
    }
}
