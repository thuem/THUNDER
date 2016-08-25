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

        vector<mat33> sr;

        mat33 rot;
        rot << 1, 0, 0,
               0, 1, 0,
               0, 0, 1;

        symmetryRotation(sr, rot, &sym);

        for (int i = 0; i < (int)sr.size(); i++)
            cout << sr[i] << endl;
    }
    catch (Error& error)
    {
        std::cout << error;
    }
}
