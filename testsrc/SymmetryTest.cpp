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
    loggerInit(argc, argv);

        Symmetry sym("C15");
        display(sym);

        std::cout << "nSymmetryElement = " << sym.nSymmetryElement() << std::endl;

        vector<mat33> sr;

        mat33 rot;
        rot << 1, 0, 0,
               0, 1, 0,
               0, 0, 1;

        symmetryRotation(sr, rot, &sym);

        std::cout << "pgOrder = " << sym.pgOrder() << std::endl;

        for (int i = 0; i < (int)sr.size(); i++)
            std::cout << sr[i] << std::endl;
}
